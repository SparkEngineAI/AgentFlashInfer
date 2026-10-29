import { useCallback, useEffect, useRef, useState } from "react";

export const useSSEStreamByTrace = (sessionId, shouldConnect) => {
    const [isConnecting, setIsConnecting] = useState(false);
    const [isConnected, setIsConnected] = useState(false);
    const [traces, setTraces] = useState({});
    const [totalMetricsEvents, setTotalMetricsEvents] = useState([]);
    const [error, setError] = useState(null);
    const eventSourceRef = useRef(null);

    // 统一缓冲区：trace_id -> Partial<Trace>  使用 Ref 避免高频触发渲染，保证同步存取
    const tracesBufferRef = useRef({});
    //
    // 1. 批量更新器：每 50ms 同步一次（更流畅）
    useEffect(() => {
        const timer = setInterval(() => {
            const buffer = tracesBufferRef.current;
            if (Object.keys(buffer).length === 0) return;

            setTraces(prev => {
                const nextTraces = { ...prev };
                for (const [traceId, updates] of Object.entries(tracesBufferRef.current)) {
                    const current = nextTraces[traceId] || {
                        durationEvent: null,
                        metricsEvents: [],
                        progressEvents: [],
                        startTime_ms: null,
                        endTime_ms: null,
                        resultEvent: null,
                        tokens: {},
                    };

                    // 1. tokens: span_id 级别合并（你的写法已正确，但需确保 buffer 写入无误）
                    const mergedTokens = { ...current.tokens };
                    if (updates.tokens) {
                        for (const [spanId, delta] of Object.entries(updates.tokens)) {
                            mergedTokens[spanId] = (mergedTokens[spanId] || '') + delta;
                        }
                    }

                    // 2. progressEvents: 数组合并
                    const mergedProgress = current.progressEvents.concat(updates.progressEvents || []);

                    // 3. metricsEvents: 数组合并
                    const mergedMetrics = current.metricsEvents.concat(updates.metricsEvents || []);

                    const newDurationEvent = updates.durationEvent !== null
                        ? updates.durationEvent
                        : current.durationEvent;
                    nextTraces[traceId] = {
                        ...current,
                        ...updates,
                        // 显式覆盖可变字段
                        tokens: mergedTokens,
                        progressEvents: mergedProgress,
                        metricsEvents: mergedMetrics,
                        durationEvent: newDurationEvent,
                        // 注意：durationEvent / resultEvent / time 字段可直接覆盖（它们是最终状态）
                    };
                }
                tracesBufferRef.current = {};
                return nextTraces;
            });
        }, 50); // 从 120ms  50ms，更流畅

        return () => clearInterval(timer);
    }, []);

    // 2. 当 shouldConnect 变化时，重置状态
    useEffect(() => {
        if (!shouldConnect) {
            setTotalMetricsEvents([]);
            tracesBufferRef.current = {};
        }
    }, [shouldConnect]);


    useEffect(() => {
        if (!shouldConnect || !sessionId) return;

        const connectSSE = async () => {
            setIsConnecting(true);
            setError(null);

            const EventSourceImpl = window.EventSource;
            const eventSourceUrl = `/api/tracestream/${sessionId}`;

            try {
                const eventSource = new EventSourceImpl(eventSourceUrl);

                eventSource.onopen = () => {
                    setIsConnecting(false);
                    setIsConnected(true);
                };

                eventSource.onmessage = (event) => {
                    try {
                        const parsedData = transformEvent(event.data);
                        if (!parsedData || !parsedData.trace_id) return;
                        const trace_id = parsedData.trace_id;

                        // 获取或初始化 trace buffer
                        let traceBuf = tracesBufferRef.current[trace_id];
                        if (!traceBuf) {
                            traceBuf = {
                                tokens: {},
                                progressEvents: [],
                                metricsEvents: [],
                                startTime_ms: null,
                                endTime_ms: null,
                                resultEvent:null,
                                durationEvent: null,
                                // 其他字段按需初始化
                            };
                            tracesBufferRef.current[trace_id] = traceBuf;
                        }

                        // 情况 A: Token 流
                        if (parsedData.span_type === "trace.span.chunk") {
                            const current = traceBuf.tokens[parsedData.span_id] || "";
                            traceBuf.tokens[parsedData.span_id] = current + (parsedData.delta || "");
                            return;
                        }

                        // 情况 B: 结构化事件
                        if (parsedData.event === 'on_trace_end') {
                            traceBuf.progressEvents.push(parsedData);
                            traceBuf.endTime_ms = parsedData.timestamp;
                        } else if (parsedData.event === 'on_trace_start') {
                            traceBuf.progressEvents.push(parsedData);
                            traceBuf.startTime_ms = parsedData.timestamp;
                        } else if (parsedData.event === 'on_span_end' && parsedData?.data?.name === 'planner_objective_task_done') {
                            traceBuf.progressEvents.push(parsedData);
                            traceBuf.resultEvent = parsedData;
                        } else if (parsedData.event === "on_status_update") {
                            // on_status_update 不进 progressEvents（和注释逻辑一致）
                            if (Array.isArray(parsedData.data) && parsedData.data.length > 0) {
                                traceBuf.durationEvent = parsedData;
                            }
                        } else {
                            // 所有其他事件（包括 generation span_end）都先进入 progressEvents
                            traceBuf.progressEvents.push(parsedData);

                            // 再额外判断是否进 metricsEvents
                            if (parsedData.span_type === "custom" && (parsedData.data?.name === 'llm_call_metric' || parsedData.data.name === 'e2e_call_metric')) {
                                traceBuf.metricsEvents.push(parsedData);
                                setTotalMetricsEvents(prev => [...prev, parsedData]);
                            } else if (parsedData.event === 'on_span_end' && parsedData.span_type === "generation") {
                                traceBuf.metricsEvents.push(parsedData);
                                // 注意：generation 不触发 setTotalMetricsEvents（和注释一致）
                            }
                        }
                    } catch (err) {
                        console.error('SSE parse error:', err);
                    }
                };

                eventSource.onerror = () => {
                    setIsConnected(false);
                    eventSource.close();
                };

                eventSourceRef.current = eventSource;
            } catch (err) {
                setIsConnecting(false);
            }
        };

        connectSSE();
        return () => eventSourceRef.current?.close();
    }, [sessionId, shouldConnect]);

    const getTraceData = useCallback((traceId) => traces[traceId] || null, [traces]);

    return { traces, getTraceData, totalMetricsEvents, isConnecting, isConnected, error, setIsConnected };
};

function transformEvent(str) {
    // 1. 去掉 "data: " 前缀（通常是6个字符）
    let jsonStr = str.startsWith('data:') ? str.slice(6).trim() : str.trim();
    // 2. 解析原始 JSON
    try {
        const rawStr = JSON.parse(jsonStr);
        const raw = JSON.parse(rawStr);

        if (raw.event === "on_trace_start" || raw.event === "on_trace_end") {
            return {
                event: raw.event,
                trace_id: raw.trace_id,
                workflow: raw.workflow,
                timestamp:raw.timestamp,
            }
        } else {
            return {
                ...raw,
                event: raw.event, // 按需重命名
                data: raw.data,
                parent_id: raw.parent_id,
                span_id: raw.span_id,
                span_type: raw.span_type,
                trace_id: raw.trace_id
            };
        }
    } catch (e) {
        console.error(e);
    }
}
