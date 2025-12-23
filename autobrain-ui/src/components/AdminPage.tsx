import { useEffect, useRef, useState } from "react";
import type { AdminResponseMessage } from "../types";

type AdminLogStatus = "ok" | "error";

interface AdminLogEntry {
  id: string;
  ts: number;
  cmd: string;
  status: AdminLogStatus;
  summary?: string;
  response?: Record<string, unknown>;
  requestId?: number | string;
}

interface AdminPageProps {
  wsUrl: string;
}

const COLORS = ["red", "green", "white", "yellow", "purple"];

const toNumber = (value: string) => {
  if (!value) return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
};

const formatTimestamp = (value: number) => {
  const date = new Date(value);
  return new Intl.DateTimeFormat(undefined, {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  }).format(date);
};

export const AdminPage = ({ wsUrl }: AdminPageProps) => {
  const socketRef = useRef<WebSocket | null>(null);
  const requestIdRef = useRef(1);
  const [connected, setConnected] = useState(false);
  const [lastError, setLastError] = useState<string | null>(null);
  const [logEntries, setLogEntries] = useState<AdminLogEntry[]>([]);

  const [flipTrackId, setFlipTrackId] = useState("");
  const [flipDelta, setFlipDelta] = useState("180");
  const [yawTrackId, setYawTrackId] = useState("");
  const [yawValue, setYawValue] = useState("");
  const [colorTrackId, setColorTrackId] = useState("");
  const [colorValue, setColorValue] = useState("");
  const [swapIdA, setSwapIdA] = useState("");
  const [swapIdB, setSwapIdB] = useState("");
  const [carCount, setCarCount] = useState("");

  const pushLog = (entry: AdminLogEntry) => {
    setLogEntries((prev) => [entry, ...prev].slice(0, 50));
  };

  const sendCommand = (cmd: string, payload: Record<string, unknown>) => {
    const socket = socketRef.current;
    if (!socket || socket.readyState !== WebSocket.OPEN) {
      setLastError("WebSocket is not connected.");
      return;
    }
    const requestId = requestIdRef.current++;
    const message = {
      type: "adminCommand",
      requestId,
      cmd,
      ...payload,
    };
    socket.send(JSON.stringify(message));
    setLastError(null);
  };

  const handleResponse = (msg: AdminResponseMessage) => {
    const ts = typeof msg.ts === "number"
      ? (msg.ts < 1e12 ? msg.ts * 1000 : msg.ts)
      : Date.now();
    const response = msg.response ?? {};
    const responseStatus =
      typeof response.status === "string" ? response.status : undefined;
    const responseMessage =
      typeof response.message === "string" ? response.message : undefined;
    const rawStatus = responseStatus || msg.status || "error";
    const status: AdminLogStatus = rawStatus === "ok" ? "ok" : "error";
    const cmd = msg.cmd ?? "unknown";
    let summary = responseMessage || msg.message || "";
    const tracks = Array.isArray(response.tracks) ? response.tracks : undefined;
    if (cmd === "list_tracks" && tracks) {
      const count = typeof response.count === "number" ? response.count : undefined;
      const size = typeof count === "number" ? count : tracks.length;
      summary = `tracks: ${size}`;
    }
    pushLog({
      id: `${msg.requestId ?? "resp"}-${ts}`,
      ts,
      cmd,
      status,
      summary,
      response,
      requestId: msg.requestId,
    });
  };

  useEffect(() => {
    let ws: WebSocket | null = null;
    let reconnectTimer: number | null = null;

    const connect = () => {
      ws = new WebSocket(wsUrl);
      socketRef.current = ws;

      ws.onopen = () => {
        setConnected(true);
        setLastError(null);
      };

      ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data) as AdminResponseMessage;
          if (msg.type === "adminResponse") {
            handleResponse(msg);
          }
        } catch (error) {
          console.warn("Admin WS: invalid message", error);
        }
      };

      ws.onclose = () => {
        setConnected(false);
        if (!reconnectTimer) {
          reconnectTimer = window.setTimeout(() => {
            reconnectTimer = null;
            connect();
          }, 2000);
        }
      };

      ws.onerror = () => {
        setConnected(false);
        ws?.close();
      };
    };

    connect();

    return () => {
      if (reconnectTimer) window.clearTimeout(reconnectTimer);
      ws?.close();
      if (socketRef.current === ws) {
        socketRef.current = null;
      }
    };
  }, [wsUrl]);

  const flipTrackIdValue = toNumber(flipTrackId);
  const flipDeltaValue = toNumber(flipDelta) ?? 180;
  const yawTrackIdValue = toNumber(yawTrackId);
  const yawValueNum = toNumber(yawValue);
  const colorTrackIdValue = toNumber(colorTrackId);
  const swapIdAValue = toNumber(swapIdA);
  const swapIdBValue = toNumber(swapIdB);
  const carCountValue = toNumber(carCount);

  const statusLabel = connected ? "Connected" : "Disconnected";
  const statusClass = connected ? "admin-status--on" : "admin-status--off";

  return (
    <main className="admin-page">
      <section className="admin-page__controls">
        <div className="panel admin-panel">
          <div className="admin-panel__header">
            <h2 className="panel__title">Command Center</h2>
            <span className={`admin-status ${statusClass}`}>{statusLabel}</span>
          </div>
          {lastError && <div className="admin-error">{lastError}</div>}
          <p className="admin-panel__hint">
            Commands are sent to the realtime server command queue via WebSocket.
          </p>
        </div>

        <div className="panel admin-panel">
          <h2 className="panel__title">Flip Yaw</h2>
          <form
            className="admin-form"
            onSubmit={(event) => {
              event.preventDefault();
              if (flipTrackIdValue === null) return;
              sendCommand("flip_yaw", {
                track_id: flipTrackIdValue,
                delta: flipDeltaValue,
              });
            }}
          >
            <label className="admin-field">
              <span>Track ID</span>
              <input
                type="number"
                min="1"
                step="1"
                value={flipTrackId}
                onChange={(event) => setFlipTrackId(event.target.value)}
                className="admin-input"
              />
            </label>
            <label className="admin-field">
              <span>Delta (deg)</span>
              <input
                type="number"
                step="1"
                value={flipDelta}
                onChange={(event) => setFlipDelta(event.target.value)}
                className="admin-input"
              />
            </label>
            <button
              type="submit"
              className="admin-button"
              disabled={flipTrackIdValue === null}
            >
              Send
            </button>
          </form>
        </div>

        <div className="panel admin-panel">
          <h2 className="panel__title">Set Yaw</h2>
          <form
            className="admin-form"
            onSubmit={(event) => {
              event.preventDefault();
              if (yawTrackIdValue === null || yawValueNum === null) return;
              sendCommand("set_yaw", {
                track_id: yawTrackIdValue,
                yaw: yawValueNum,
              });
            }}
          >
            <label className="admin-field">
              <span>Track ID</span>
              <input
                type="number"
                min="1"
                step="1"
                value={yawTrackId}
                onChange={(event) => setYawTrackId(event.target.value)}
                className="admin-input"
              />
            </label>
            <label className="admin-field">
              <span>Yaw (deg)</span>
              <input
                type="number"
                step="1"
                value={yawValue}
                onChange={(event) => setYawValue(event.target.value)}
                className="admin-input"
              />
            </label>
            <button
              type="submit"
              className="admin-button"
              disabled={yawTrackIdValue === null || yawValueNum === null}
            >
              Send
            </button>
          </form>
        </div>

        <div className="panel admin-panel">
          <h2 className="panel__title">Set Color</h2>
          <form
            className="admin-form"
            onSubmit={(event) => {
              event.preventDefault();
              if (colorTrackIdValue === null) return;
              const payload: Record<string, unknown> = {
                track_id: colorTrackIdValue,
              };
              if (colorValue) {
                payload.color = colorValue;
              } else {
                payload.color = null;
              }
              sendCommand("set_color", payload);
            }}
          >
            <label className="admin-field">
              <span>Track ID</span>
              <input
                type="number"
                min="1"
                step="1"
                value={colorTrackId}
                onChange={(event) => setColorTrackId(event.target.value)}
                className="admin-input"
              />
            </label>
            <label className="admin-field">
              <span>Color</span>
              <select
                value={colorValue}
                onChange={(event) => setColorValue(event.target.value)}
                className="admin-select"
              >
                <option value="">Auto</option>
                {COLORS.map((color) => (
                  <option key={color} value={color}>
                    {color}
                  </option>
                ))}
              </select>
            </label>
            <button
              type="submit"
              className="admin-button"
              disabled={colorTrackIdValue === null}
            >
              Send
            </button>
          </form>
        </div>

        <div className="panel admin-panel">
          <h2 className="panel__title">Swap IDs</h2>
          <form
            className="admin-form"
            onSubmit={(event) => {
              event.preventDefault();
              if (swapIdAValue === null || swapIdBValue === null) return;
              sendCommand("swap_ids", {
                track_id_a: swapIdAValue,
                track_id_b: swapIdBValue,
              });
            }}
          >
            <label className="admin-field">
              <span>Track ID A</span>
              <input
                type="number"
                min="1"
                step="1"
                value={swapIdA}
                onChange={(event) => setSwapIdA(event.target.value)}
                className="admin-input"
              />
            </label>
            <label className="admin-field">
              <span>Track ID B</span>
              <input
                type="number"
                min="1"
                step="1"
                value={swapIdB}
                onChange={(event) => setSwapIdB(event.target.value)}
                className="admin-input"
              />
            </label>
            <button
              type="submit"
              className="admin-button"
              disabled={swapIdAValue === null || swapIdBValue === null}
            >
              Send
            </button>
          </form>
        </div>

        <div className="panel admin-panel">
          <h2 className="panel__title">Car Count</h2>
          <form
            className="admin-form"
            onSubmit={(event) => {
              event.preventDefault();
              if (carCountValue === null) return;
              sendCommand("set_car_count", { car_count: carCountValue });
            }}
          >
            <label className="admin-field">
              <span>Car count (1-5)</span>
              <input
                type="number"
                min="1"
                max="5"
                step="1"
                value={carCount}
                onChange={(event) => setCarCount(event.target.value)}
                className="admin-input"
              />
            </label>
            <button
              type="submit"
              className="admin-button"
              disabled={carCountValue === null}
            >
              Send
            </button>
          </form>
        </div>

        <div className="panel admin-panel">
          <h2 className="panel__title">List Tracks</h2>
          <button
            type="button"
            className="admin-button"
            onClick={() => sendCommand("list_tracks", {})}
          >
            Fetch
          </button>
        </div>
      </section>

      <section className="panel admin-log">
        <div className="admin-log__header">
          <h2 className="panel__title">Command Log</h2>
          <span className="admin-log__count">{logEntries.length} entries</span>
        </div>
        {logEntries.length === 0 && (
          <div className="panel__empty">No responses yet.</div>
        )}
        {logEntries.map((entry) => (
          <div key={entry.id} className="admin-log__entry">
            <div className="admin-log__meta">
              <span className={`admin-log__status admin-log__status--${entry.status}`}>
                {entry.status.toUpperCase()}
              </span>
              <span className="admin-log__cmd">{entry.cmd}</span>
              {entry.requestId !== undefined && (
                <span className="admin-log__id">#{entry.requestId}</span>
              )}
              <span className="admin-log__time">{formatTimestamp(entry.ts)}</span>
            </div>
            {entry.summary && <div className="admin-log__summary">{entry.summary}</div>}
            {entry.response && (
              <pre className="admin-log__payload">
                {JSON.stringify(entry.response, null, 2)}
              </pre>
            )}
          </div>
        ))}
      </section>
    </main>
  );
};
