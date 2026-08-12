
// Package httpapi exposes the gateway REST + WebSocket surface for the dashboard.
package httpapi

import (
	"encoding/json"
	"log"
	"net/http"
	"time"

	"github.com/google/uuid"
	"github.com/gorilla/websocket"

	"github.com/pd241008/BlackICE-Mesh/gateway-service/internal/broker"
	"github.com/pd241008/BlackICE-Mesh/gateway-service/internal/store"
	"github.com/pd241008/BlackICE-Mesh/gateway-service/internal/ws"
)

// Server ties together the broker, store and websocket hub behind an HTTP mux.
type Server struct {
	broker *broker.Broker
	store  *store.Store
	hub    *ws.Hub
	mux    *http.ServeMux
}

var upgrader = websocket.Upgrader{
	ReadBufferSize:  1024,
	WriteBufferSize: 4096,
	CheckOrigin:     func(r *http.Request) bool { return true },
}

// New wires all HTTP routes.
func New(b *broker.Broker, s *store.Store, h *ws.Hub) *Server {
	srv := &Server{broker: b, store: s, hub: h, mux: http.NewServeMux()}

	srv.mux.HandleFunc("GET /api/v1/health", srv.health)
	srv.mux.HandleFunc("POST /api/v1/jobs", srv.createJob)
	srv.mux.HandleFunc("GET /api/v1/results", srv.recentResults)
	srv.mux.HandleFunc("GET /ws", srv.stream)
	return srv
}

// Handler exposes the underlying mux.
func (s *Server) Handler() http.Handler {
	return s.mux
}

func (s *Server) health(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, envelope(true, map[string]any{"status": "ok", "service": "blackice-gateway"}, nil))
}

type jobRequest struct {
	Type    string         `json:"type"`
	Payload map[string]any `json:"payload"`
}

func (s *Server) createJob(w http.ResponseWriter, r *http.Request) {
	var req jobRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, envelope(false, nil, err.Error()))
		return
	}
	if req.Type == "" {
		writeJSON(w, http.StatusBadRequest, envelope(false, nil, "missing job type"))
		return
	}

	job := broker.Job{
		JobID:   uuid.NewString(),
		Type:    req.Type,
		Payload: req.Payload,
	}
	if err := s.broker.PublishJob(r.Context(), job); err != nil {
		log.Printf("httpapi: publish failed: %v", err)
		writeJSON(w, http.StatusServiceUnavailable, envelope(false, nil, "broker unavailable"))
		return
	}

	writeJSON(w, http.StatusAccepted, envelope(true, map[string]any{"job_id": job.JobID, "type": job.Type, "status": "queued"}, nil))
}

func (s *Server) recentResults(w http.ResponseWriter, r *http.Request) {
	results, err := s.store.RecentResults(r.Context(), 50)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, envelope(false, nil, err.Error()))
		return
	}
	writeJSON(w, http.StatusOK, envelope(true, map[string]any{"results": results}, nil))
}

func (s *Server) stream(w http.ResponseWriter, r *http.Request) {
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Printf("ws: upgrade failed: %v", err)
		return
	}
	s.hub.Register(conn)
	defer s.hub.Unregister(conn)

	conn.SetReadLimit(1024)
	conn.SetReadDeadline(time.Now().Add(90 * time.Second))
	conn.SetPongHandler(func(string) error {
		return conn.SetReadDeadline(time.Now().Add(90 * time.Second))
	})
	for {
		if _, _, err := conn.ReadMessage(); err != nil {
			return
		}
	}
}

// envelope wraps every response in the Design-Dungeons unified API shape.
func envelope(success bool, data any, errMsg any) map[string]any {
	return map[string]any{"success": success, "data": data, "error": errMsg}
}

func writeJSON(w http.ResponseWriter, status int, body any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(body)
}
