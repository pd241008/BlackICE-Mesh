// Package broker wraps RabbitMQ connectivity for the BlackICE-Mesh gateway.
package broker

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"time"

	amqp "github.com/rabbitmq/amqp091-go"
)

const (
	JobQueue        = "ml.jobs"
	ResultsExchange = "ml.results"
)

// Broker owns the AMQP channel used to dispatch ML jobs and ingest results.
type Broker struct {
	conn *amqp.Connection
	ch   *amqp.Channel
}

// Connect establishes a durable AMQP connection with reconnect backoff.
func Connect(url string) (*Broker, error) {
	var conn *amqp.Connection
	var err error
	for i := 0; i < 30; i++ {
		conn, err = amqp.DialConfig(url, amqp.Config{
			Heartbeat: 10 * time.Second,
		})
		if err == nil {
			break
		}
		log.Printf("broker: dial failed (%v), retrying in 2s", err)
		time.Sleep(2 * time.Second)
	}
	if err != nil {
		return nil, fmt.Errorf("broker: %w", err)
	}

	ch, err := conn.Channel()
	if err != nil {
		conn.Close()
		return nil, fmt.Errorf("broker channel: %w", err)
	}

	if _, err := ch.QueueDeclare(JobQueue, true, false, false, false, nil); err != nil {
		return nil, fmt.Errorf("queue declare: %w", err)
	}
	if err := ch.ExchangeDeclare(ResultsExchange, "direct", true, false, false, false, nil); err != nil {
		return nil, fmt.Errorf("exchange declare: %w", err)
	}

	log.Printf("broker: connected to %s", url)
	return &Broker{conn: conn, ch: ch}, nil
}

// Close tears down the AMQP connection.
func (b *Broker) Close() error {
	if b.ch != nil {
		b.ch.Close()
	}
	return b.conn.Close()
}

// Job is the message envelope the gateway publishes to the ML worker.
type Job struct {
	JobID   string         `json:"job_id"`
	Type    string         `json:"type"`
	Payload map[string]any `json:"payload"`
}

// PublishJob dispatches an ML job to the durable worker queue.
func (b *Broker) PublishJob(ctx context.Context, job Job) error {
	body, err := json.Marshal(job)
	if err != nil {
		return err
	}
	return b.ch.PublishWithContext(ctx, "", JobQueue, false, false, amqp.Publishing{
		ContentType:  "application/json",
		DeliveryMode: amqp.Persistent,
		Body:         body,
	})
}

// ResultHandler is invoked for every telemetry message emitted by the ML worker.
type ResultHandler func(message []byte)

// ConsumeResults subscribes to the results exchange until ctx is cancelled.
func (b *Broker) ConsumeResults(ctx context.Context, exchange string, handler ResultHandler) error {
	queue, err := b.ch.QueueDeclare("", false, true, true, false, nil)
	if err != nil {
		return fmt.Errorf("declare result queue: %w", err)
	}
	if err := b.ch.QueueBind(queue.Name, "result.#", exchange, false, nil); err != nil {
		return fmt.Errorf("bind result queue: %w", err)
	}

	msgs, err := b.ch.Consume(queue.Name, "", true, false, false, false, nil)
	if err != nil {
		return fmt.Errorf("consume results: %w", err)
	}

	for {
		select {
		case <-ctx.Done():
			return nil
		case msg, ok := <-msgs:
			if !ok {
				return fmt.Errorf("broker: results channel closed")
			}
			handler(msg.Body)
		}
	}
}
