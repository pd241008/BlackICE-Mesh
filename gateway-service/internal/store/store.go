// Package store persists ML telemetry into PostgreSQL.
package store

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
)

// Store is a thin Postgres-backed repository for experiment results.
type Store struct {
	pool *pgxpool.Pool
}

// Connect dials Postgres and applies the telemetry schema.
func Connect(ctx context.Context, dsn string) (*Store, error) {
	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		return nil, fmt.Errorf("store connect: %w", err)
	}

	schema := `
CREATE TABLE IF NOT EXISTS results (
    id          BIGSERIAL PRIMARY KEY,
    job_id      TEXT NOT NULL,
    type        TEXT NOT NULL,
    status      TEXT NOT NULL,
    payload     JSONB NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS results_created_at_idx ON results (created_at DESC);
`
	if _, err := pool.Exec(ctx, schema); err != nil {
		return nil, fmt.Errorf("store migrate: %w", err)
	}

	log.Println("store: postgres ready")
	return &Store{pool: pool}, nil
}

// Close releases the connection pool.
func (s *Store) Close() {
	s.pool.Close()
}

// SaveResult writes one telemetry message produced by the ML worker.
func (s *Store) SaveResult(ctx context.Context, jobID, kind, status string, payload any) error {
	body, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	_, err = s.pool.Exec(ctx,
		`INSERT INTO results (job_id, type, status, payload) VALUES ($1, $2, $3, $4)`,
		jobID, kind, status, body)
	return err
}

// RecentResults returns the latest N telemetry records for the dashboard.
func (s *Store) RecentResults(ctx context.Context, limit int) ([]map[string]any, error) {
	rows, err := s.pool.Query(ctx, `
		SELECT job_id, type, status, payload, created_at
		FROM results ORDER BY created_at DESC LIMIT $1`, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var out []map[string]any
	for rows.Next() {
		var (
			jobID, kind, status string
			payload             []byte
			createdAt           time.Time
		)
		if err := rows.Scan(&jobID, &kind, &status, &payload, &createdAt); err != nil {
			return nil, err
		}
		var pl any
		_ = json.Unmarshal(payload, &pl)
		out = append(out, map[string]any{
			"job_id": jobID, "type": kind, "status": status,
			"payload": pl, "created_at": createdAt,
		})
	}
	return out, rows.Err()
}
