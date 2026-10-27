// Job updates from the server's WebSocket, reconnecting with back-off if the connection drops.

import type { Job } from "./api";

/** Connect to /api/events. Returns a function that disconnects for good. */
export function connectJobEvents(onJob: (job: Job) => void, onConnected: (connected: boolean) => void): () => void {
  let socket: WebSocket | null = null;
  let stopped = false;
  let delay = 500;
  let retry: number | undefined;

  const connect = () => {
    const protocol = window.location.protocol === "https:" ? "wss" : "ws";
    socket = new WebSocket(`${protocol}://${window.location.host}/api/events`);
    socket.onopen = () => {
      delay = 500;
      onConnected(true);
    };
    socket.onmessage = (event: MessageEvent<string>) => {
      const message = JSON.parse(event.data) as { type: string; job?: Job };
      if (message.type === "job" && message.job) onJob(message.job);
    };
    socket.onclose = () => {
      onConnected(false);
      if (stopped) return;
      retry = window.setTimeout(connect, delay);
      delay = Math.min(delay * 2, 10_000);
    };
  };

  connect();
  return () => {
    stopped = true;
    window.clearTimeout(retry);
    socket?.close();
  };
}
