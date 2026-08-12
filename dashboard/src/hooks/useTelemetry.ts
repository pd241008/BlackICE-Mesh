"use client";

import { useEffect, useState } from "react";
import { GATEWAY_URL } from "@/lib/api";
import type { TelemetryResult } from "@/lib/types";

const MAX_EVENTS = 50;

function websocketUrl(): string {
  return `${GATEWAY_URL.replace(/^http/, "ws")}/ws`;
}

export function useTelemetry() {
  const [events, setEvents] = useState<TelemetryResult[]>([]);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    let socket: WebSocket | null = null;
    let closed = false;
    let retry = 0;

    const connect = () => {
      socket = new WebSocket(websocketUrl());
      socket.onopen = () => {
        setConnected(true);
        retry = 0;
      };
      socket.onclose = () => {
        setConnected(false);
        if (!closed) {
          retry = Math.min(retry + 1, 5);
          setTimeout(connect, 2000 * retry);
        }
      };
      socket.onerror = () => socket?.close();
      socket.onmessage = (ev: MessageEvent<string>) => {
        try {
          const msg = JSON.parse(ev.data) as TelemetryResult;
          if (msg.job_id) {
            setEvents((prev) => [msg, ...prev].slice(0, MAX_EVENTS));
          }
        } catch {
          // malformed frames are dropped silently
        }
      };
    };

    connect();
    return () => {
      closed = true;
      socket?.close();
    };
  }, []);

  return { events, connected };
}
