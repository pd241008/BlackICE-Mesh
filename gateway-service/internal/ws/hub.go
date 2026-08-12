// Package ws implements a realtime fan-out hub for dashboard telemetry.
package ws

import (
	"log"
	"sync"

	"github.com/gorilla/websocket"
)

// Hub fans telemetry messages out to every subscribed dashboard client.
type Hub struct {
	mu      sync.RWMutex
	clients map[*websocket.Conn]bool
}

// NewHub creates an empty hub.
func NewHub() *Hub {
	return &Hub{clients: make(map[*websocket.Conn]bool)}
}

// Register attaches a websocket client to the hub.
func (h *Hub) Register(conn *websocket.Conn) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.clients[conn] = true
	log.Println("ws: client connected")
}

// Unregister detaches a client and closes its socket.
func (h *Hub) Unregister(conn *websocket.Conn) {
	h.mu.Lock()
	defer h.mu.Unlock()
	if _, ok := h.clients[conn]; ok {
		delete(h.clients, conn)
		conn.Close()
		log.Println("ws: client disconnected")
	}
}

// Broadcast pushes a raw JSON message to all clients.
func (h *Hub) Broadcast(message []byte) {
	h.mu.RLock()
	defer h.mu.RUnlock()
	for conn := range h.clients {
		if err := conn.WriteMessage(websocket.TextMessage, message); err != nil {
			conn.Close()
			delete(h.clients, conn)
		}
	}
}
