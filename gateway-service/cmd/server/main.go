// Command server is the BlackICE-Mesh gateway: it brokers RabbitMQ traffic,
// persists telemetry to PostgreSQL and streams results to the dashboard.
package main

import (
	"context"
	"encoding/json"
	"log"
	"net/http"
	"os"
	"time"

	"github.com/pd241008/BlackICE-Mesh/gateway-service/internal/broker"
	"github.com/pd241008/BlackICE-Mesh/gateway-service/internal/httpapi"
	"github.com/pd241008/BlackICE-Mesh/gateway-service/internal/store"
	"github.com/pd241008/BlackICE-Mesh/gateway-service/internal/ws"
)

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func main() {
	amqpURL := envOr("AMQP_URL", "amqp://blackice:blackice@localhost:5672/")
	pgDSN := envOr(
		"DATABASE_URL",
		"postgres://blackice:blackice@localhost:5432/blackice?sslmode=disable",
	)
	addr := envOr("GATEWAY_ADDR", ":8080")

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	brk, err := broker.Connect(amqpURL)
	if err != nil {
		log.Fatalf("gateway: %v", err)
	}
	defer brk.Close()

	st, err := store.Connect(ctx, pgDSN)
	if err != nil {
		log.Fatalf("gateway: %v", err)
	}
	defer st.Close()

	hub := ws.NewHub()

	// Realtime fan-out: broker -> websocket hub + postgres persistence.
	go func() {
		if err := brk.ConsumeResults(ctx, broker.ResultsExchange, func(msg []byte) {
			log.Printf("gateway: result <- %s", msg)
			hub.Broadcast(msg)

			var envelope map[string]any
			if err := json.Unmarshal(msg, &envelope); err != nil {
				log.Printf("gateway: cannot decode result: %v", err)
				return
			}
			jobID, _ := envelope["job_id"].(string)
			kind, _ := envelope["type"].(string)
			status := "ok"
			if envelope["error"] != nil {
				status = "error"
			}
			if err := st.SaveResult(ctx, jobID, kind, status, envelope); err != nil {
				log.Printf("gateway: persist failed: %v", err)
			}
		}); err != nil {
			log.Fatalf("gateway: results consumer: %v", err)
		}
	}()

	srv := &http.Server{
		Addr:              addr,
		Handler:           httpapi.New(brk, st, hub).Handler(),
		ReadHeaderTimeout: 10 * time.Second,
	}

	log.Printf("gateway: listening on %s", addr)
	if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatalf("gateway: %v", err)
	}
}
