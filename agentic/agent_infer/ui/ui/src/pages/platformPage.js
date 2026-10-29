import React, {useState, useEffect} from 'react';
import { Play, Zap, BarChart3, Layers, X, ChevronLeft, Copy, Check, Activity, Users, Cpu, TrendingUp, Grid3x3, Square, AlertCircle ,  Sun, Moon, Settings,} from 'lucide-react';
import SessionContainer from "../components/sessionContainer";
import {parseIfJson} from "../utils/formatUtil";
import {useGlobal} from "../context/globalContext";
import ConfigModal from "../components/ConfigModal";

const MultiAgentPlatform = ({userToken}) => {
    const [tasks, setTasks] = useState([]);
    const [concurrencyConfig, setConcurrencyConfig] = useState({ default: 1 });
    const [userInput, setUserInput] = useState('');
    const [expandedTasks, setExpandedTasks] = useState(new Set());
    const [selectedEventFlow, setSelectedEventFlow] = useState(null);
    const [selectedThread, setSelectedThread] = useState(null);
    const [shouldConnectSSE, setShouldConnectSSE] = useState(false);
    const [taskToEvents, setTaskToEvents] = useState({});

    const [realTimeStats, setRealTimeStats] = useState({
        activeUsers: 0,
        totalConcurrentTasks: 0,
        peakConcurrency: 0,
        concurrentTasks:0
        // avgLatency: 0
    });
    const [llmMetrics, setLlmMetrics] = useState([]);

    // 配置相关状态
    const [configModalOpen, setConfigModalOpen] = useState(false);
    const [windowConfig, setWindowConfig] = useState({});

    const { isBlank, setIsBlank } = useGlobal();

    // 👇 新增的 useEffect
    useEffect(() => {
        const fetchStats = async () => {
            try {
                const res = await fetch('/api/stats');
                if (!res.ok) {
                    console.error('Failed to fetch stats:', res.status);
                    return;
                }
                const data = await res.json();

                const { total_running_tasks, tasks_per_user } = data;
                const currentUserTasks = tasks_per_user[userToken] ?? 0;
                const activeUsers = Object.keys(tasks_per_user).length;
                const totalConcurrentTasks = total_running_tasks;

                setRealTimeStats(prev => ({
                    activeUsers,
                    totalConcurrentTasks,
                    concurrentTasks: currentUserTasks,
                    peakConcurrency: Math.max(prev.peakConcurrency, totalConcurrentTasks)
                }));
            } catch (error) {
                console.error('Error fetching real-time stats:', error);
            }
        };

        fetchStats();
        const intervalId = setInterval(fetchStats, 3000);
        return () => clearInterval(intervalId);
    }, [userToken]);

    // 加载窗口配置
    useEffect(() => {
        const fetchWindowConfig = async () => {
            try {
                const res = await fetch(`/api/window/config/${userToken}`);
                if (res.ok) {
                    const data = await res.json();
                    if (Object.keys(data).length > 0) {
                        setWindowConfig(data);
                    }
                }
            } catch (error) {
                console.error('Failed to load window config:', error);
            }
        };
        fetchWindowConfig();
    }, [userToken]);

    // 配置变更处理
    const handleConfigChange = (newConfig) => {
        setWindowConfig(newConfig);
    };

    // 提交任务
    const addTask = async () => {
        if (!userInput.trim()) return;
        const numThreads = concurrencyConfig.default || 1;
        const res = await fetch('/api/submit', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                user_input: userInput,
                agent_name: 'assistant',
                concurrency: numThreads,
                user_token:userToken
            }),
        });
        if (!res.ok) {
            throw new Error(`HTTP ${res.status}`);
        }
        const data = await res.json();
        const newSessionId = data.session_id;
        if (!newSessionId) {
            throw new Error('No session_id returned');
        }
        const traceIds = data.trace_ids;
        const taskId = newSessionId;
        // todo session_id 和对应的几个trace_id是需要通过接口拿到， 这里session_id = task_id


        const threads = traceIds.map((traceId,index) => {
            return {
                id: traceId,
                threadIndex: index,
                agentGroups: {},
                events:{},
                startTime_ms: null,
                name: `Thread ` + index,
                endTime_ms:null
            }
            }
        )

        const newTask = {
            id: taskId,
            traceIds,
            input: userInput,
            concurrency: concurrencyConfig.default || 1,
            status: 'running',
            progress: 0,
            events: [], // 存储事件流
            threads: threads, // 按trace_id分组的线程数据
            summary: null,
            startTime: new Date(),
            stopResult: null, // 停止操作结果
        };


        setTasks(prev => [...prev, newTask]);
        setUserInput('');
        setSelectedThread(0); // 使用traceId作为线程标识
        setShouldConnectSSE(true);

    };

    const toggleTaskExpand = (taskId) => {
        setExpandedTasks(prev => {
            const newSet = new Set(prev);
            if (newSet.has(taskId)) newSet.delete(taskId);
            else newSet.add(taskId);
            return newSet;
        });
    };
    // 1. 抽象主题配置
    const theme = {
        bg: isBlank ? '#0a0a14' : '#f8f9fa',
        panelBg: isBlank ? '#0f172a' : '#fff',
        textPrimary: isBlank ? '#f1f5f9' : '#0a0a0a',
        textSecondary: isBlank ? '#64748b' : '#525252',
        border: isBlank ? '#334155' : '#e5e7eb',
        subtleBorder: isBlank ? '#1e293b' : '#e5e7eb',
        accent: isBlank ? '#10b981' : '#06b6d4',
        inputBg: isBlank ? 'rgba(15, 23, 42, 0.6)' : '#fff',
        cardRadius: isBlank ? '8px' : '12px',
        scrollbarThumb: isBlank ? '#334155' : '#d1d5db'
    };
    // 2. 辅助渲染组件：StatSection
    const renderStats = () => (
        <div style={{ padding: '20px 24px', borderBottom: `1px solid ${theme.border}`, backgroundColor: theme.panelBg }}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '16px' }}>
                {isBlank ? (
                    <>
                        <StatCard icon="⚡" label="Curent Concurrent Tasks" value={realTimeStats.concurrentTasks} unit="tasks" />
                        <StatCard icon="📈" label="Peak Concurrency" value={realTimeStats.peakConcurrency} unit="max" />
                        <StatCard icon="👥" label="Active Users" value={realTimeStats.activeUsers} unit="" />
                        <StatCard icon="⚡" label="Total Concurrent Tasks" value={realTimeStats.totalConcurrentTasks} unit="tasks" />
                    </>
                ) : (
                    <>
                        <StatCardWhite icon="⚡" label="Curent Concurrent Tasks" value={realTimeStats.concurrentTasks} unit="tasks" />
                        <StatCardWhite icon="📈" label="Peak Concurrency" value={realTimeStats.peakConcurrency} unit="max" />
                        <StatCardWhite icon="👥" label="Active Users" value={realTimeStats.activeUsers} unit="" />
                        <StatCardWhite icon="⚡" label="Total Concurrent Tasks" value={realTimeStats.totalConcurrentTasks} unit="tasks" />
                    </>
                )}
            </div>
        </div>
    );

    // 3. 辅助渲染组件：LLM Metrics
    const renderLLMMetrics = () => {
        if (llmMetrics.length === 0) return null;
        const latest = llmMetrics[llmMetrics.length - 1]?.data?.metrics;
        const metricsList = [
            { label: 'TTFT (ms)', key: 'ttft_ms' },
            { label: 'TPOT (ms)', key: 'tpot_ms' },
            { label: 'P99 ITL (ms)', key: 'p99_itl_ms' },
            { label: 'TOKEN COUNT', key: 'token_count' }
        ];

        return (
            <div style={{ padding: '20px 24px', borderBottom: `1px solid ${theme.border}`, backgroundColor: theme.panelBg }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '12px' }}>
                    <Cpu size={14} style={{ color: '#f59e0b' }} />
                    <span style={{ fontSize: '12px', fontWeight: isBlank ? '300' : '700', letterSpacing: '0.05em', color: isBlank ? '#fbbf24' : '#0a0a0a' }}>LLM PERFORMANCE METRICS</span>
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '16px' }}>
                    {metricsList.map(m => (
                        <div key={m.key} style={{
                            background: isBlank ? 'rgba(245, 158, 11, 0.1)' : '#fff7ed',
                            border: `1px solid ${isBlank ? '#f59e0b' : '#fed7aa'}`,
                            borderRadius: theme.cardRadius, padding: '12px'
                        }}>
                            <div style={{ fontSize: '11px', color: isBlank ? '#fbbf24' : '#9a3412', fontWeight: isBlank ? '300' : '700', marginBottom: '4px' }}>{m.label}</div>
                            <div style={{ fontSize: isBlank ? '16px' : '18px', fontWeight: isBlank ? '300' : '700', color: isBlank ? '#f59e0b' : '#0a0a0a' }}>
                                {m.key === 'token_count' ? (latest?.[m.key] || 0) : (latest?.[m.key]?.toFixed(2) || 0)}
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        );
    };
    return (
        <div style={{ minHeight: '100vh', backgroundColor: theme.bg, color: theme.textPrimary, fontFamily: 'Geist Mono, monospace', display: 'flex', flexDirection: 'column' }}>
            {/* Header */}
            <div style={{
                borderBottom: `1px solid ${theme.border}`,
                backgroundColor: theme.panelBg,
                padding: '16px 24px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                boxShadow: isBlank ? 'none' : '0 1px 3px rgba(0,0,0,0.05)',
                zIndex: 40,
                transition: 'all 0.3s ease'
            }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                    <div style={{
                        width: '8px',
                        height: '8px',
                        borderRadius: '50%',
                        backgroundColor: theme.accent,
                        animation: 'pulse 2s infinite'
                    }}></div>
                    <h1 style={{
                        fontSize: '18px',
                        fontWeight: isBlank ? '300' : '600',
                        letterSpacing: '0.05em',
                        margin: 0,
                        color: theme.textPrimary
                    }}>
                        MULTI-AGENT RESEARCH PLATFORM
                    </h1>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
                    {/* 窗口配置按钮 */}
                    <button
                        onClick={() => setConfigModalOpen(true)}
                        style={{
                            display: 'flex',
                            alignItems: 'center',
                            gap: '8px',
                            padding: '6px 12px',
                            borderRadius: '6px',
                            border: `1px solid ${theme.border}`,
                            backgroundColor: windowConfig._is_default !== true
                                ? (isBlank ? 'rgba(16, 185, 129, 0.3)' : 'rgba(34, 197, 195, 0.15)')
                                : (isBlank ? 'rgba(51, 65, 85, 0.3)' : '#f9fafb'),
                            color: windowConfig._is_default !== true
                                ? (isBlank ? '#10b981' : '#0891b2')
                                : theme.textSecondary,
                            cursor: 'pointer',
                            fontSize: '11px',
                            fontWeight: '600',
                            transition: 'all 0.2s ease',
                            outline: 'none',
                            letterSpacing: '0.5px'
                        }}
                        onMouseEnter={(e) => {
                            e.currentTarget.style.backgroundColor = isBlank ? 'rgba(51, 65, 85, 0.5)' : '#f3f4f6';
                            e.currentTarget.style.color = theme.textPrimary;
                        }}
                        onMouseLeave={(e) => {
                            e.currentTarget.style.backgroundColor = windowConfig._is_default !== true
                                ? (isBlank ? 'rgba(16, 185, 129, 0.3)' : 'rgba(34, 197, 195, 0.15)')
                                : (isBlank ? 'rgba(51, 65, 85, 0.3)' : '#f9fafb');
                            e.currentTarget.style.color = windowConfig._is_default !== true
                                ? (isBlank ? '#10b981' : '#0891b2')
                                : theme.textSecondary;
                        }}
                    >
                        <Settings size={14} />
                        <span>{windowConfig._is_default !== true ? '配置 (已设置)' : '配置'}</span>
                    </button>

                    {/* 主题切换按钮 */}
                    <button
                        onClick={() => setIsBlank(!isBlank)}
                        style={{
                            display: 'flex',
                            alignItems: 'center',
                            gap: '8px',
                            padding: '6px 12px',
                            borderRadius: '6px',
                            border: `1px solid ${theme.border}`,
                            backgroundColor: isBlank ? 'rgba(51, 65, 85, 0.3)' : '#f9fafb',
                            color: theme.textSecondary,
                            cursor: 'pointer',
                            fontSize: '11px',
                            fontWeight: '600',
                            transition: 'all 0.2s ease',
                            outline: 'none',
                            letterSpacing: '0.5px'
                        }}
                        onMouseEnter={(e) => {
                            e.currentTarget.style.backgroundColor = isBlank ? 'rgba(51, 65, 85, 0.5)' : '#f3f4f6';
                            e.currentTarget.style.color = theme.textPrimary;
                        }}
                        onMouseLeave={(e) => {
                            e.currentTarget.style.backgroundColor = isBlank ? 'rgba(51, 65, 85, 0.3)' : '#f9fafb';
                            e.currentTarget.style.color = theme.textSecondary;
                        }}
                    >
                        {isBlank ? (
                            <>
                                <Sun size={14} />
                                <span>LIGHT MODE</span>
                            </>
                        ) : (
                            <>
                                <Moon size={14} />
                                <span>DARK MODE</span>
                            </>
                        )}
                    </button>

                    {/* 原有的版本号或其他占位 */}
                    <div style={{ fontSize: '11px', color: theme.textSecondary }}></div>
                </div>
            </div>
            <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
                <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'auto' }}>
                    {renderStats()}

                    {/* User Input Section */}
                    <div style={{ padding: '20px 24px', borderBottom: `1px solid ${theme.border}`, backgroundColor: theme.panelBg }}>
                        <div style={{
                            background: isBlank
                                ? 'linear-gradient(135deg, rgba(34, 197, 195, 0.1), rgba(59, 130, 246, 0.1))'
                                : 'linear-gradient(135deg, rgba(6, 182, 212, 0.08), rgba(34, 197, 195, 0.08))',
                            border: `1px solid ${isBlank ? '#06b6d4' : '#cffafe'}`,
                            borderRadius: theme.cardRadius, padding: '16px'
                        }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px' }}>
                                <Zap size={14} style={{ color: '#06b6d4' }} />
                                <span style={{ fontSize: '12px', fontWeight: isBlank ? '300' : '700', letterSpacing: isBlank ? '0.05em' : '0.8px', color: isBlank ? '#22d3ee' : '#0a0a0a' }}>EXECUTE QUERY</span>
                            </div>
                            <div style={{ display: 'flex', gap: '8px' }}>
                                <input
                                    type="text"
                                    value={userInput}
                                    onChange={(e) => setUserInput(e.target.value)}
                                    onKeyPress={(e) => e.key === 'Enter' && addTask()}
                                    placeholder="Enter your research query..."
                                    style={{ flex: 1, backgroundColor: theme.inputBg, border: `1px solid ${theme.border}`, borderRadius: isBlank ? '6px' : '8px', padding: '10px 12px', fontSize: isBlank ? '12px' : '13px', color: theme.textPrimary, outline: 'none' }}
                                />
                                <button
                                    onClick={addTask}
                                    disabled={!userInput.trim()}
                                    style={{ backgroundColor: userInput.trim() ? (isBlank ? '#0891b2' : '#06b6d4') : (isBlank ? '#334155' : '#d1d5db'), border: 'none', borderRadius: isBlank ? '6px' : '8px', padding: '10px 16px', color: '#fff', cursor: userInput.trim() ? 'pointer' : 'not-allowed', display: 'flex', alignItems: 'center', gap: '6px', fontWeight: isBlank ? '300' : '600' }}
                                >
                                    <Play size={14} /> SUBMIT
                                </button>
                            </div>
                        </div>
                    </div>

                    {/* Concurrency Config Section */}
                    <div style={{ padding: '20px 24px', borderBottom: `1px solid ${theme.border}`, backgroundColor: theme.panelBg }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '12px' }}>
                            <Zap size={14} style={{ color: isBlank ? '#10b981' : '#06b6d4' }} />
                            <span style={{ fontSize: '12px', fontWeight: isBlank ? '300' : '700', letterSpacing: isBlank ? '0.05em' : '0.8px', color: isBlank ? '#cbd5e1' : '#0a0a0a' }}>CONCURRENCY CONFIG</span>
                        </div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                            {[1, 4, 8, 16].map(num => (
                                <button
                                    key={num}
                                    onClick={() => setConcurrencyConfig({ default: num })}
                                    style={{
                                        padding: '8px 12px', borderRadius: isBlank ? '6px' : '8px', fontSize: isBlank ? '12px' : '13px', fontWeight: isBlank ? '300' : '600', transition: 'all 0.2s', flex: '1', cursor: 'pointer',
                                        backgroundColor: concurrencyConfig.default === num ? (isBlank ? 'rgba(16, 185, 129, 0.3)' : '#cffafe') : (isBlank ? 'rgba(51, 65, 85, 0.3)' : '#f3f4f6'),
                                        border: concurrencyConfig.default === num ? `1.5px solid ${isBlank ? '#10b981' : '#06b6d4'}` : `1px solid ${theme.border}`,
                                        color: concurrencyConfig.default === num ? (isBlank ? '#10b981' : '#0a0a0a') : theme.textSecondary
                                    }}
                                >
                                    ×{num}
                                </button>
                            ))}
                            <div style={{ width: '30px' }} />
                            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', flex: '1.2' }}>
                                <span style={{ fontSize: '11px', color: theme.textSecondary, fontWeight: isBlank ? '300' : '600', whiteSpace: 'nowrap' }}>CUSTOM:</span>
                                <input
                                    type="number" min="1" max="16"
                                    value={concurrencyConfig.default}
                                    onChange={(e) => setConcurrencyConfig({ default: Math.max(1, Math.min(16, parseInt(e.target.value) || 1)) })}
                                    style={{ width: '100%', padding: '8px 10px', borderRadius: isBlank ? '6px' : '8px', fontSize: '12px', fontWeight: isBlank ? '300' : '600', backgroundColor: theme.inputBg, border: `1px solid ${theme.border}`, color: theme.textPrimary, outline: 'none' }}
                                />
                            </div>
                        </div>
                    </div>

                    {renderLLMMetrics()}

                    {/* Tasks List Container */}
                    <div style={{ flex: 1, overflow: 'auto', padding: '20px 24px', backgroundColor: isBlank ? 'transparent' : '#f8f9fa' }}>
                        {tasks.length === 0 ? (
                            <div style={{ textAlign: 'center', paddingTop: '60px', color: theme.textSecondary }}>
                                <div style={{ fontSize: '48px', marginBottom: '16px' }}>🚀</div>
                                <div style={{ fontSize: '14px', fontWeight: isBlank ? '300' : '500' }}>Enter a query and submit to begin analysis</div>
                            </div>
                        ) : (
                            <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                                {tasks.map(task => (
                                    <SessionContainer
                                        key={task.id}
                                        task={task}
                                        expandedTasks={expandedTasks}
                                        selectedEventFlow={selectedEventFlow}
                                        toggleTaskExpand={toggleTaskExpand}
                                        setSelectedEventFlow={setSelectedEventFlow}
                                        shouldConnect={shouldConnectSSE}
                                        setTasks={setTasks}
                                    />
                                ))}
                            </div>
                        )}
                    </div>
                </div>
            </div>

            {selectedEventFlow && (
                <EventDetailPanel event={selectedEventFlow} onClose={() => setSelectedEventFlow(null)} />
            )}

            <ConfigModal
                isOpen={configModalOpen}
                onClose={() => setConfigModalOpen(false)}
                userToken={userToken}
                config={windowConfig}
                onConfigChange={handleConfigChange}
            />

            <style>{`
                @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: ${isBlank ? '0.5' : '0.6'}; } }
                ${!isBlank ? '* { scrollbar-width: thin; scrollbar-color: #d1d5db #f9fafb; }' : ''}
                *::-webkit-scrollbar { width: 6px; height: 6px; }
                *::-webkit-scrollbar-thumb { background: ${theme.scrollbarThumb}; border-radius: 3px; }
                *::-webkit-scrollbar-track { background: ${isBlank ? 'transparent' : '#f9fafb'}; }
            `}</style>
        </div>
    );
};

const EventDetailPanel = ({ event, onClose }) => {
    const isTool = event.hasOwnProperty('input') && event.hasOwnProperty('output');
    const isGeneration = event.hasOwnProperty('messages') && event.hasOwnProperty('response');
    const { isBlank } = useGlobal();

    // --- 样式变量定义 ---
    const theme = {
        panelBg: isBlank ? '#0f172a' : '#ffffff',
        headerBorder: isBlank ? '#1e293b' : '#f1f5f9',
        panelBorder: isBlank ? '#334155' : '#e5e7eb',
        titleColor: isBlank ? '#f1f5f9' : '#0a0a0a',
        subtitleColor: isBlank ? '#64748b' : '#737373',
        blockBg: isBlank ? '#1a1f3a' : '#f8fafc',
        blockBorder: isBlank ? '#334155' : '#e2e8f0',
        contentColor: isBlank ? '#cbd5e1' : '#1e293b',
        shadow: isBlank ? 'none' : '-4px 0 20px rgba(0,0,0,0.05)',
        // 角色配色（浅色模式稍微加深以保证对比度）
        blue: isBlank ? '#4A90E2' : '#2563eb',
        green: isBlank ? '#10B981' : '#059669',
        amber: isBlank ? '#F59E0B' : '#d97706',
    };

    const formatString = (str) => {
        if (typeof str !== 'string') return str || '';
        try {
            return JSON.parse(`"${str}"`);
        } catch (e) {
            return str
                .replace(/\\n/g, '\n')
                .replace(/\\r/g, '\r')
                .replace(/\\t/g, '\t')
                .replace(/\\"/g, '"')
                .replace(/\\\\/g, '\\');
        }
    };

    const parseJsonArray = (jsonStr) => {
        if (!jsonStr) return [];
        if(typeof jsonStr !== 'string') return jsonStr;
        try {
            const parsed = JSON.parse(jsonStr);
            return Array.isArray(parsed) ? parsed : [];
        } catch (e) {
            return [];
        }
    };

    const renderMessageBlock = (msg, index) => {
        if (!msg || typeof msg !== 'object') return null;
        const role = (msg.role || '').toUpperCase();
        const content = formatString(msg.content || '');

        let roleColor = theme.blue;
        if (role === 'ASSISTANT') roleColor = theme.green;
        else if (role === 'SYSTEM') roleColor = theme.amber;

        return (
            <div key={index} style={{
                marginBottom: '12px',
                backgroundColor: theme.blockBg,
                border: `1px solid ${theme.blockBorder}`,
                borderRadius: '8px',
                padding: '12px',
                fontSize: isBlank ? '11px' : '12px',
                color: theme.contentColor,
                fontFamily: isBlank ? 'monospace' : 'inherit',
                whiteSpace: 'pre-wrap',
                wordBreak: 'break-word',
                lineHeight: isBlank ? 'normal' : '1.6'
            }}>
                <div style={{
                    color: roleColor,
                    fontWeight: 'bold',
                    marginBottom: '6px',
                    fontSize: '10px',
                    letterSpacing: isBlank ? 'normal' : '0.5px'
                }}>
                    {role}
                </div>
                <div style={{ fontFamily: 'monospace' }}>{content}</div>
            </div>
        );
    };

    const messages = isGeneration ? parseJsonArray(event.messages) : [];
    const response = isGeneration ? parseJsonArray(event.response) : [];

    return (
        <div style={{
            position: 'fixed',
            right: 0,
            top: 0,
            height: '100vh',
            width: isBlank ? '380px' : '400px',
            backgroundColor: theme.panelBg,
            borderLeft: `1px solid ${theme.panelBorder}`,
            display: 'flex',
            flexDirection: 'column',
            zIndex: 1000,
            boxShadow: theme.shadow,
            transition: 'all 0.3s ease'
        }}>
            {/* Header */}
            <div style={{
                borderBottom: `1px solid ${theme.headerBorder}`,
                padding: isBlank ? '16px 20px' : '20px 24px',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center'
            }}>
                <div>
                    <div style={{ fontSize: isBlank ? '13px' : '15px', color: theme.titleColor, fontWeight: '700' }}>
                        {isTool ? (event.name || 'Tool Detail') : 'Generation'}
                    </div>
                    <div style={{ fontSize: '11px', color: theme.subtitleColor, marginTop: '2px', fontWeight: isBlank ? 'normal' : '600' }}>
                        {isTool ? 'TOOL EXECUTION DETAILS' : 'LLM GENERATION DETAILS'}
                    </div>
                </div>
                <button
                    onClick={onClose}
                    style={{ background: 'none', border: 'none', color: theme.subtitleColor, cursor: 'pointer' }}
                >
                    <X size={isBlank ? 18 : 20} />
                </button>
            </div>

            <div style={{ flex: 1, overflow: 'auto', padding: isBlank ? '20px' : '24px' }}>
                {/* TOOL: INPUT / OUTPUT */}
                {isTool && (
                    <>
                        <SectionLabel title="INPUT" color={theme.blue} isBlank={isBlank} />
                        <DataBox content={event.input} theme={theme} format={formatString} isBlank={isBlank} />

                        <SectionLabel title="OUTPUT" color={theme.green} isBlank={isBlank} />
                        <DataBox content={event.output} theme={theme} format={formatString} isBlank={isBlank} />
                    </>
                )}

                {/* GENERATION: MESSAGES + RESPONSE */}
                {isGeneration && (
                    <>
                        {messages.length > 0 && (
                            <div style={{ marginBottom: '16px' }}>
                                <SectionLabel title="MESSAGES" color={theme.blue} isBlank={isBlank} />
                                {messages.map((msg, idx) => renderMessageBlock(msg, idx))}
                            </div>
                        )}
                        {response.length > 0 && (
                            <div>
                                <SectionLabel title="RESPONSE" color={theme.green} isBlank={isBlank} />
                                {response.map((msg, idx) => renderMessageBlock(msg, idx))}
                            </div>
                        )}
                    </>
                )}
            </div>
        </div>
    );
};

// --- 辅助小型组件（封装重复样式） ---

const SectionLabel = ({ title, color, isBlank }) => (
    <div style={{
        color: color,
        fontSize: '11px',
        fontWeight: '700',
        letterSpacing: '0.05em',
        marginBottom: '8px',
        display: 'flex',
        alignItems: 'center',
        gap: '6px'
    }}>
        {!isBlank && <div style={{ width: '3px', height: '10px', backgroundColor: color, borderRadius: '2px' }} />}
        {title}
    </div>
);

const DataBox = ({ content, theme, format, isBlank }) => (
    <div style={{
        backgroundColor: theme.blockBg,
        border: `1px solid ${theme.blockBorder}`,
        borderRadius: isBlank ? '6px' : '8px',
        padding: isBlank ? '12px' : '14px',
        fontSize: isBlank ? '11px' : '12px',
        color: theme.contentColor,
        fontFamily: 'monospace',
        whiteSpace: 'pre-wrap',
        wordBreak: 'break-word',
        marginBottom: '20px',
        lineHeight: isBlank ? 'normal' : '1.6'
    }}>
        {format(content)}
    </div>
);

const StatCard = ({ icon, label, value, unit }) => (
    <div style={{ background: 'rgba(51, 65, 85, 0.3)', border: '1px solid #334155', borderRadius: '8px', padding: '12px', display: 'flex', alignItems: 'center', gap: '12px' }}>
        <div style={{ fontSize: '24px' }}>{icon}</div>
        <div>
            <div style={{ fontSize: '11px', color: '#64748b' }}>{label}</div>
            <div style={{ fontSize: '16px', fontWeight: '300', color: '#f1f5f9', marginTop: '2px' }}>
                {value} <span style={{ fontSize: '11px', color: '#64748b' }}>{unit}</span>
            </div>
        </div>
    </div>
);

const StatCardWhite = ({ icon, label, value, unit }) => (
    <div style={{
        background: '#fff',
        border: '1px solid #e5e7eb',
        borderRadius: '12px',
        padding: '12px',
        display: 'flex',
        alignItems: 'center',
        gap: '12px',
        boxShadow: '0 1px 2px rgba(0,0,0,0.02)' // 增加极轻微的阴影提升质感
    }}>
        <div style={{ fontSize: '24px' }}>{icon}</div>
        <div>
            <div style={{
                fontSize: '11px',
                color: '#525252',
                fontWeight: '600',
                letterSpacing: '0.5px',
                textTransform: 'uppercase' // 让标签更具设计感
            }}>
                {label}
            </div>
            <div style={{
                fontSize: '18px',
                fontWeight: '700',
                color: '#0a0a0a',
                marginTop: '2px'
            }}>
                {value}
                {unit && (
                    <span style={{ fontSize: '11px', color: '#737373', fontWeight: '500', marginLeft: '4px' }}>
                        {unit}
                    </span>
                )}
            </div>
        </div>
    </div>
);
export default MultiAgentPlatform;
