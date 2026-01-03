import { useEffect, useRef, useState } from "react";
import type { AdminResponseMessage, CarId } from "../types";

type NoticeTone = "ok" | "error";

interface Notice {
  tone: NoticeTone;
  text: string;
}

interface AdminControlsPanelProps {
  wsUrl: string;
  selectedCarId: CarId | null;
}

const COLORS = ["red", "green", "yellow", "purple", "white"];

const toNumber = (value: string) => {
  if (!value) return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
};

const parseTrackId = (carId: CarId | null) => {
  if (!carId) return null;
  const cleaned = carId.toString().trim().replace("car-", "");
  const parsed = Number(cleaned);
  return Number.isFinite(parsed) ? parsed : null;
};

export const AdminControlsPanel = ({
  wsUrl,
  selectedCarId,
}: AdminControlsPanelProps) => {
  const socketRef = useRef<WebSocket | null>(null);
  const requestIdRef = useRef(1);
  const [connected, setConnected] = useState(false);
  const [notice, setNotice] = useState<Notice | null>(null);

  const [flipDelta, setFlipDelta] = useState("180");
  const [yawValue, setYawValue] = useState("");
  const [colorValue, setColorValue] = useState("");
  const [swapIdA, setSwapIdA] = useState("");
  const [swapIdB, setSwapIdB] = useState("");
  const [carCount, setCarCount] = useState("");

  const trackId = parseTrackId(selectedCarId);
  const flipDeltaValue = toNumber(flipDelta) ?? 180;
  const yawValueNum = toNumber(yawValue);
  const swapIdAValue = toNumber(swapIdA);
  const swapIdBValue = toNumber(swapIdB);
  const carCountValue = toNumber(carCount);

  const sendCommand = (cmd: string, payload: Record<string, unknown>) => {
    const socket = socketRef.current;
    if (!socket || socket.readyState !== WebSocket.OPEN) {
      setNotice({ tone: "error", text: "WebSocket disconnected." });
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
  };

  const handleResponse = (msg: AdminResponseMessage) => {
    if (msg.type !== "adminResponse") return;
    const response = msg.response ?? {};
    const responseStatus =
      typeof response.status === "string" ? response.status : undefined;
    const responseMessage =
      typeof response.message === "string" ? response.message : undefined;
    const statusRaw = responseStatus || msg.status || "error";
    const tone: NoticeTone = statusRaw === "ok" ? "ok" : "error";
    const messageText =
      responseMessage || msg.message || (tone === "ok" ? "ok" : "error");
    const cmd = msg.cmd ? `${msg.cmd}: ` : "";
    setNotice({ tone, text: `${cmd}${messageText}` });
  };

  useEffect(() => {
    let ws: WebSocket | null = null;
    let reconnectTimer: number | null = null;

    const connect = () => {
      ws = new WebSocket(wsUrl);
      socketRef.current = ws;

      ws.onopen = () => {
        setConnected(true);
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

  const statusLabel = connected ? "Connected" : "Disconnected";
  const statusClass = connected ? "admin-status--on" : "admin-status--off";
  const selectedLabel = trackId !== null ? `${trackId}` : "Select a car";

  return (
    <>
      <section className="panel admin-actions">
        <div className="admin-panel__header">
          <h2 className="panel__title">Admin Controls</h2>
          <span className={`admin-status ${statusClass}`}>{statusLabel}</span>
        </div>
        <div className="admin-selected">
          <span className="admin-selected__label">Selected</span>
          <span className="admin-selected__value">{selectedLabel}</span>
        </div>
        {notice && (
          <div className={`admin-notice admin-notice--${notice.tone}`}>
            {notice.text}
          </div>
        )}
        <div className="admin-actions__grid">
          <form
            className="admin-inline-form"
            onSubmit={(event) => {
              event.preventDefault();
              if (trackId === null) return;
              sendCommand("flip_yaw", {
                track_id: trackId,
                delta: flipDeltaValue,
              });
            }}
          >
            <span className="admin-inline-label">Flip yaw</span>
            <div className="admin-inline-row">
              <input
                type="number"
                step="1"
                value={flipDelta}
                onChange={(event) => setFlipDelta(event.target.value)}
                className="admin-input admin-input--compact"
              />
              <button
                type="submit"
                className="admin-button admin-button--compact"
                disabled={trackId === null || !connected}
              >
                Send
              </button>
            </div>
          </form>

          <form
            className="admin-inline-form"
            onSubmit={(event) => {
              event.preventDefault();
              if (trackId === null || yawValueNum === null) return;
              sendCommand("set_yaw", {
                track_id: trackId,
                yaw: yawValueNum,
              });
            }}
          >
            <span className="admin-inline-label">Set yaw</span>
            <div className="admin-inline-row">
              <input
                type="number"
                step="1"
                value={yawValue}
                onChange={(event) => setYawValue(event.target.value)}
                className="admin-input admin-input--compact"
              />
              <button
                type="submit"
                className="admin-button admin-button--compact"
                disabled={trackId === null || yawValueNum === null || !connected}
              >
                Send
              </button>
            </div>
          </form>

          <form
            className="admin-inline-form"
            onSubmit={(event) => {
              event.preventDefault();
              if (trackId === null) return;
              const payload: Record<string, unknown> = {
                track_id: trackId,
              };
              payload.color = colorValue ? colorValue : null;
              sendCommand("set_color", payload);
            }}
          >
            <span className="admin-inline-label">Set color</span>
            <div className="admin-inline-row">
              <select
                value={colorValue}
                onChange={(event) => setColorValue(event.target.value)}
                className="admin-select admin-select--compact"
              >
                <option value="">Auto</option>
                {COLORS.map((color) => (
                  <option key={color} value={color}>
                    {color}
                  </option>
                ))}
              </select>
              <button
                type="submit"
                className="admin-button admin-button--compact"
                disabled={trackId === null || !connected}
              >
                Send
              </button>
            </div>
          </form>
        </div>
      </section>

      <section className="panel admin-compact">
        <h2 className="panel__title">Quick Admin</h2>
        <form
          className="admin-inline-form admin-inline-form--row"
          onSubmit={(event) => {
            event.preventDefault();
            if (swapIdAValue === null || swapIdBValue === null) return;
            sendCommand("swap_ids", {
              track_id_a: swapIdAValue,
              track_id_b: swapIdBValue,
            });
          }}
        >
          <span className="admin-inline-label">Swap IDs</span>
          <div className="admin-inline-row">
            <input
              type="number"
              min="1"
              step="1"
              value={swapIdA}
              onChange={(event) => setSwapIdA(event.target.value)}
              className="admin-input admin-input--compact"
              placeholder="A"
            />
            <input
              type="number"
              min="1"
              step="1"
              value={swapIdB}
              onChange={(event) => setSwapIdB(event.target.value)}
              className="admin-input admin-input--compact"
              placeholder="B"
            />
            <button
              type="submit"
              className="admin-button admin-button--compact"
              disabled={
                swapIdAValue === null || swapIdBValue === null || !connected
              }
            >
              Swap
            </button>
          </div>
        </form>

        <form
          className="admin-inline-form admin-inline-form--row"
          onSubmit={(event) => {
            event.preventDefault();
            if (carCountValue === null) return;
            sendCommand("set_car_count", { car_count: carCountValue });
          }}
        >
          <span className="admin-inline-label">Car count</span>
          <div className="admin-inline-row">
            <input
              type="number"
              min="1"
              max="5"
              step="1"
              value={carCount}
              onChange={(event) => setCarCount(event.target.value)}
              className="admin-input admin-input--compact"
              placeholder="1-5"
            />
            <button
              type="submit"
              className="admin-button admin-button--compact"
              disabled={carCountValue === null || !connected}
            >
              Set
            </button>
          </div>
        </form>
      </section>
    </>
  );
};
