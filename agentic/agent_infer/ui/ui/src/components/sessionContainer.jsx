// src/components/SessionContainer.jsx
import React, {useState, useEffect, useCallback, useMemo, useRef, useLayoutEffect} from 'react';
import {AlertCircle, BarChart3, Grid3x3, Layers, X, Square, Cpu, Maximize2, Minimize2, Eye, EyeOff} from 'lucide-react';
import TraceVisualization from './traceVisualization';
import {useSSEStreamByTrace} from "../hook/useSSEStreamByTrace";
import {useGlobal} from "../context/globalContext";
import DurationTable from "./durationTable";
import MetricsStreamingPanel from "./metricsStreamingPanel";
import {extractFinalAnswer} from "../utils/formatUtil";

export default function SessionContainer({  task,
                                             expandedTasks,
                                             selectedEventFlow,
                                             toggleTaskExpand,
                                             setSelectedEventFlow,
                                             shouldConnect,
                                             setTasks
                                         }) {
    const { traces,getTraceData, totalMetricsEvents, isConnecting, isConnected ,error, setIsConnected} = useSSEStreamByTrace(task.id, shouldConnect);
    const [selectedThreadIndex, setSelectedThreadIndex] = useState(0);
    const selectedThread = task.threads[selectedThreadIndex];
    const metricsEvents = getTraceData(selectedThread?.id)? getTraceData(selectedThread?.id).metricsEvents : [];
    const progressEvents = getTraceData(selectedThread?.id)?getTraceData(selectedThread?.id).progressEvents:[];
    const durationEvent = getTraceData(selectedThread?.id)?getTraceData(selectedThread?.id)?.durationEvent:null;
    const tokenStreamMap = getTraceData(selectedThread?.id)?getTraceData(selectedThread?.id)?.tokens:{};
    const [isMaximized, setIsMaximized] = useState(false);
    const [showFullTrace, setShowFullTrace] = useState(false);
    const [viewMode, setViewMode] = useState('timeline');
    const [totalViewMode, setTotalViewMode] = useState('timeline');
    const [finalSummary, setFinalSummary] = useState(null);
    const { isBlank, setIsBlank } = useGlobal();

    const totalDurationEvent = useMemo(() => {
        return mergeLatencyFromTracesPure(traces);
    }, [traces]); // 依赖 traces 引用

    // 判断任务是否结束：traces是否所有trace endTime_ms都不为空
    useEffect(() => {
        // 过滤掉整个session的trace
        const traceValues = Object.values(traces);
        const completedCount = traceValues.filter(
            trace =>  trace.endTime_ms != null && trace.endTime_ms !== 0
        ).length;
        const allTracesHaveEndTime = traceValues.length > 0 &&
            completedCount >= task.threads.length

        if (allTracesHaveEndTime || task.status!=='running') {
            // 统计：仅基于 endTime_ms 是否存在
            let completed = completedCount;
            let failed = task.threads.length - completedCount;

            // 如果所有线程都有 endTime_ms，则任务完成
            const taskId = task.id;
            if (task.status === 'running') {
                setTasks(prev =>
                    prev.map(t => (t.id === taskId ? { ...t, status: 'completed' } : t))
                );
            }
            setIsConnected(false);
            generateFinalSummary(completed, failed); // 只传两个参数
            return;

        }
    }, [traces, task.status,task.id]);


    const generateFinalSummary = (completed, stopped) => {
        if (finalSummary) return;
        const threadSummaries = task.threads.map(thread => {
            const traceData = getTraceData(thread?.id);
            const start = traceData?.startTime_ms ?? null;
            const end = traceData?.endTime_ms || Date.now();
            const status = traceData?.endTime_ms ?'completed':'stopped';
            const finalAnswer = extractFinalAnswer(traceData?.resultEvent);
            let durationText = '未知';
            let durationSeconds = null;

            if (start != null && end != null && end >= start) {
                durationSeconds = (end - start) / 1000; // 转为秒
                if (durationSeconds < 1) {
                    durationText = '<1 秒';
                } else {
                    durationText = `${durationSeconds.toFixed(1)} 秒`;
                }
            }
            // 判断状态
            let error = null;
            return {
                threadId: thread.id,
                name: thread.name,
                status,
                duration: durationSeconds,
                durationText,
                finalAnswer,
                // error,
            };
        });

        // 总耗时：取所有 thread 中的最大耗时（或平均？这里按最大）
        const maxDuration = Math.max(...threadSummaries
            .map(t => t.duration || 0)
            .filter(d => d > 0)
        );
        let totalDurationText = '未知';
        if (maxDuration > 0) {
            if (maxDuration < 1) {
                totalDurationText = '<1 秒';
            } else {
                totalDurationText = `${maxDuration.toFixed(1)} 秒`;
            }
        }

        // 构建文本摘要
        let overallStatusText = '';
        if (completed === task.threads.length) {
            overallStatusText = '✅ 所有线程执行成功';
        } else {
            overallStatusText = '❌ 部分线程执行失败';
        }

        let summaryContent = `${overallStatusText}

📊 线程执行统计：
• 总线程数: ${task.threads.length}
• 成功: ${completed}
• 失败: ${stopped}
⏱️ 最大耗时: ${totalDurationText}
`;
        // 添加每个线程详情
        threadSummaries.forEach((ts, idx) => {
            let statusEmoji = '';
            switch (ts.status) {
                case 'completed': statusEmoji = '✅'; break;
                case 'stopped': statusEmoji = '❌'; break;
                case 'running': statusEmoji = '⏹️'; break;
                // case 'interrupted': statusEmoji = '⚠️'; break;
                default: statusEmoji = '🔄';
            }

            summaryContent += `\n${statusEmoji} 线程 #${idx} (${ts.name})
   状态: ${ts.status}
   耗时: ${ts.durationText}`;

            if (ts.error) {
                summaryContent += `\n   错误: ${ts.error}`;
            }

            if (ts.finalAnswer) {
                summaryContent += `\n   最终答案:\n${ts.finalAnswer}`;
            }
            summaryContent += '\n';
        });

        setFinalSummary({
            title: '📋 任务最终执行总结',
            content: summaryContent.trim()
        });
    };

    const getText = (status) => {
        switch (status) {
            case 'completed': return '已完成';
            case 'stopped': return '已中断';
            default: return '处理中...';
        }
    };
    // 根据状态决定样式和文本
    const getBarBgColor = (status) => {
        if (status === 'completed') return '#14b8a6'; // 成功绿色
        if (status === 'stopped') return '#ef4444';   // 中断红色
        return 'rgba(51, 65, 85, 0.2)';               // 默认灰色
    };

    // 停止任务
    const stopTask =  (taskId) => {
        fetch(`/api/stop/${taskId}`, {
            method: 'GET',
        })
            .then(response => {
                let success =true
                if (!response.ok) {
                    console.error(`HTTP error! status: ${response.status}`);
                    success=false
                }
                setTasks(prev => {
                    return prev.map(task => {
                        if (task.id === taskId) {
                            let updatedTask = { ...task };

                            if (success) {
                                const updatedThreads = task.threads?.map(thread => ({
                                    ...thread,
                                    endtime_ms: Date.now()
                                })) || [];

                                updatedTask = {
                                    ...updatedTask,
                                    threads: updatedThreads,
                                    stopResult: 'stopped',
                                    status: 'stopped'
                                };
                            } else {
                                updatedTask = {
                                    ...updatedTask,
                                    stopResult: 'failed'
                                    // status 保持原样
                                };
                            }

                            return updatedTask;
                        }
                        return task;
                    });
                });
                return response.json();
            })
            .then(data => {
                // console.log('Success:', data);
            })
            .catch(error => {
                console.error('Error:', error);
            });
    };
// 1. 严格映射两套模式的样式差异
    const t = {
        containerBorder: isBlank ? '1px solid #334155' : '1px solid #e5e7eb',
        containerRadius: isBlank ? '8px' : '12px',
        containerBg: isBlank ? 'rgba(15, 23, 42, 0.5)' : '#fff',
        containerShadow: isBlank ? 'none' : '0 2px 4px rgba(0,0,0,0.04)',
        headerBg: isBlank ? 'rgba(30, 41, 59, 0.5)' : '#fff',

        // 文字样式
        inputFontSize: isBlank ? '13px' : '14px',
        inputColor: isBlank ? '#f1f5f9' : '#0a0a0a',
        inputWeight: isBlank ? '400' : '600',
        subTextColor: isBlank ? '#64748b' : '#737373',
        subTextWeight: isBlank ? '400' : '500',

        // 进度条与状态
        progressTrack: isBlank ? getBarBgColor(task.status) : '#f3f4f6',
        progressBorder: isBlank ? 'none' : '1px solid #e5e7eb',
        progressGradient: isBlank ? 'linear-gradient(90deg, transparent, #10b981, #14b8a6, transparent)' : 'linear-gradient(90deg, transparent, #06b6d4, #22d3ee, transparent)',
        statusTextColor: isBlank ? '#64748b' : '#525252',
        statusTextWeight: isBlank ? '400' : '600',

        // 停止按钮
        stopBtnBg: isBlank ? '#f8fafc' : '#fee2e2',
        stopBtnBorder: isBlank ? '1px solid #e2e8f0' : '1px solid #fecaca',
        stopBtnText: isBlank ? '#64748b' : '#dc2626',
        stopBtnPadding: isBlank ? '2px 6px' : '4px 10px',
        stopBtnRadius: isBlank ? '4px' : '6px',

        // 展开区域
        expandedBg: isBlank ? 'rgba(10, 10, 20, 0.8)' : '#f9fafb',
        expandedBorder: isBlank ? '1px solid #1e293b' : '1px solid #e5e7eb',
        dashboardHeaderBorder: isBlank ? '1px solid rgba(51, 65, 85, 0.5)' : '1px solid #e5e7eb',
        accentColor: isBlank ? '#3b82f6' : '#06b6d4',

        // 关键帧差异
        animStart: isBlank ? '-50%' : '-100%',
        animEnd: isBlank ? '150%' : '250%'
    };

    // 2. 状态区域公共逻辑
    const renderStatusArea = () => (
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <style>{`
                @keyframes indeterminate-loading {
                    0% { transform: translateX(${t.animStart}); }
                    100% { transform: translateX(${t.animEnd}); }
                }
            `}</style>

            <div style={{
                width: '120px', height: '6px', backgroundColor: t.progressTrack,
                borderRadius: '3px', overflow: 'hidden', position: 'relative', border: t.progressBorder
            }}>
                {task.status === 'running' && (
                    <div style={{
                        position: 'absolute', top: 0, left: 0, height: '100%', width: '40%',
                        background: t.progressGradient, animation: 'indeterminate-loading 1.5s infinite ease-in-out'
                    }} />
                )}
                {(task.status === 'completed' || task.status === 'aborted') && (
                    <div style={{
                        width: '100%', height: '100%',
                        backgroundColor: isBlank ? getBarBgColor(task.status) : (task.status === 'completed' ? '#10b981' : '#ef4444')
                    }} />
                )}
            </div>

            <div style={{ fontSize: '11px', color: t.statusTextColor, minWidth: '50px', textAlign: 'right', fontWeight: t.statusTextWeight }}>
                {getText(task.status)}
            </div>

            {task.status === 'running' && (
                <button
                    onClick={(e) => { e.stopPropagation(); stopTask(task.id); }}
                    style={{
                        fontSize: isBlank ? '10px' : '11px', padding: t.stopBtnPadding, background: t.stopBtnBg,
                        border: t.stopBtnBorder, borderRadius: t.stopBtnRadius, color: t.stopBtnText,
                        cursor: 'pointer', fontWeight: t.statusTextWeight, whiteSpace: 'nowrap'
                    }}
                >
                    停止
                </button>
            )}

            {task.stopResult === 'failed' && (
                <div style={{ display: 'flex', alignItems: 'center', gap: '4px', color: isBlank ? '#f87171' : '#ef4444' }}>
                    <AlertCircle size={12} />
                    <span style={{ fontSize: isBlank ? '10px' : '11px', fontWeight: t.statusTextWeight }}>Stop failed</span>
                </div>
            )}
        </div>
    );

    const maximizedStyles = isMaximized ? {
        position: 'fixed',
        top: 0,
        left: 0,
        width: '100vw',
        height: '100vh',
        zIndex: 900,
        margin: 0,
        borderRadius: 0,
        backgroundColor: isBlank ? '#0f172a' : '#fff'
    } : {};
    return (

        <div key={task.id} style={{
            ...maximizedStyles, // 注入全屏样式
            border: t.containerBorder,
            borderRadius: isMaximized ? 0 : t.containerRadius,
            overflow: 'hidden',
            backgroundColor: t.containerBg,
            backdropFilter: isBlank ? 'blur(10px)' : 'none',
            boxShadow: isMaximized ? 'none' : t.containerShadow,
            marginBottom: (isBlank || isMaximized) ? '0' : '16px',
            display: 'flex',
            flexDirection: 'column' // 确保内部可以纵向伸缩
        }}>

            {/* Header */}
            <div
                onClick={() => toggleTaskExpand(task.id)}
                style={{ padding: '16px', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'space-between', backgroundColor: t.headerBg, transition: 'background-color 0.2s' }}
            >
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flex: 1 }}>
                    <span style={{ fontSize: '14px', color: expandedTasks.has(task.id) ? (isBlank ? '#10b981' : '#06b6d4') : (isBlank ? '#64748b' : '#a3a3a3'), fontWeight: isBlank ? '400' : '700' }}>
                        {expandedTasks.has(task.id) ? '▼' : '▶'}
                    </span>
                    <div>
                        <div style={{ fontSize: t.inputFontSize, color: t.inputColor, fontWeight: t.inputWeight }}>{task.input}</div>
                        <div style={{ fontSize: '11px', color: t.subTextColor, marginTop: '4px', fontWeight: t.subTextWeight }}>
                            {task.startTime.toLocaleTimeString()} • {task.concurrency} threads
                        </div>
                    </div>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                    {renderStatusArea()}
                    <button
                        onClick={(e) => { e.stopPropagation(); setIsMaximized(!isMaximized); }}
                        style={{ background: 'none', border: 'none', cursor: 'pointer', color: t.subTextColor }}
                    >
                        {isMaximized ? <Minimize2 size={16} /> : <Maximize2 size={16} />}
                    </button>
                </div>
            </div>

            {/* Expanded Content */}
            {expandedTasks.has(task.id) && (
                <div style={{ borderTop: t.expandedBorder, padding: '20px', backgroundColor: t.expandedBg }}>
                    {/* Dashboard Header */}
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '20px', paddingBottom: '16px', borderBottom: t.dashboardHeaderBorder }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                            <Layers size={14} style={{ color: isBlank ? '#3b82f6' : '#06b6d4' }} />
                            <span style={{ fontSize: '12px', fontWeight: isBlank ? '300' : '700', letterSpacing: isBlank ? '0.05em' : '0.8px', color: isBlank ? '#cbd5e1' : '#0a0a0a' }}>
                                MONITORING DASHBOARD
                            </span>
                        </div>

                        {/* Total View Toggle */}
                        <div style={{ display: 'flex', gap: '4px', background: isBlank ? 'rgba(30, 41, 59, 0.6)' : '#f3f4f6', padding: '4px', borderRadius: isBlank ? '6px' : '8px', border: isBlank ? '1px solid #334155' : '1px solid #d1d5db' }}>
                            {[
                                { id: 'timeline', label: isBlank ? 'Threads' : 'Threads', icon: <Grid3x3 size={14} />, color: isBlank ? '#60a5fa' : '#06b6d4', activeBg: isBlank ? 'rgba(59, 130, 246, 0.3)' : '#fff' },
                                { id: 'table', label: isBlank ? 'Total Duration' : 'Duration', icon: <BarChart3 size={14} />, color: isBlank ? '#34d399' : '#10b981', activeBg: isBlank ? 'rgba(16, 185, 129, 0.3)' : '#fff' },
                                { id: 'metrics', label: isBlank ? 'Total Metrics' : 'Metrics', icon: <Cpu size={14} />, color: isBlank ? '#fbbf24' : '#f59e0b', activeBg: isBlank ? 'rgba(245, 158, 11, 0.3)' : '#fff' }
                            ].map(mode => (
                                <button
                                    key={mode.id}
                                    onClick={() => setTotalViewMode(mode.id)}
                                    style={{
                                        background: totalViewMode === mode.id ? mode.activeBg : 'transparent',
                                        border: totalViewMode === mode.id ? (isBlank ? `1px solid ${mode.color.replace('60a5fa', '3b82f6').replace('34d399', '10b981').replace('fbbf24', 'f59e0b')}` : '1px solid #d1d5db') : '1px solid transparent',
                                        borderRadius: '6px', padding: '6px 12px', cursor: 'pointer',
                                        color: totalViewMode === mode.id ? mode.color : (isBlank ? '#94a3b8' : '#737373'),
                                        display: 'flex', alignItems: 'center', gap: '6px', fontSize: '11px', fontWeight: isBlank ? '500' : '700', transition: 'all 0.2s ease',
                                        boxShadow: (!isBlank && totalViewMode === mode.id) ? '0 1px 2px rgba(0,0,0,0.05)' : 'none'
                                    }}
                                >
                                    {mode.icon}
                                    <span>{mode.label}</span>
                                </button>
                            ))}
                        </div>
                    </div>

                    {totalViewMode === 'timeline' && (
                        <>
                            {/* Thread Selector */}
                            <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '20px' }}>
                                <span style={{ fontSize: '11px', color: isBlank ? '#94a3b8' : '#525252', fontWeight: isBlank ? '500' : '700' }}>SELECT THREAD:</span>
                                <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
                                    {task.threads.map(thread => (
                                        <button
                                            key={thread.threadIndex}
                                            onClick={() => setSelectedThreadIndex(thread.threadIndex)}
                                            style={{
                                                padding: '6px 12px', fontSize: '11px', borderRadius: isBlank ? '4px' : '6px',
                                                backgroundColor: selectedThreadIndex === thread.threadIndex ? (isBlank ? 'rgba(59, 130, 246, 0.3)' : '#cffafe') : (isBlank ? 'rgba(51, 65, 85, 0.3)' : '#fff'),
                                                border: selectedThreadIndex === thread.threadIndex ? (isBlank ? '1px solid #3b82f6' : '1.5px solid #06b6d4') : (isBlank ? '1px solid #475569' : '1px solid #d1d5db'),
                                                color: selectedThreadIndex === thread.threadIndex ? (isBlank ? '#60a5fa' : '#0a0a0a') : (isBlank ? '#94a3b8' : '#525252'),
                                                cursor: 'pointer', transition: 'all 0.2s ease', fontWeight: isBlank ? '400' : '600'
                                            }}
                                        >
                                            {thread.name}
                                        </button>
                                    ))}
                                </div>
                            </div>

                            {/* Thread View Area */}
                            {(selectedThreadIndex === 0 || selectedThreadIndex) && selectedThread && (
                                <div style={{
                                    background: isBlank ? 'rgba(15, 23, 42, 0.6)' : '#fff',
                                    border: isBlank ? '1px solid #334155' : '1px solid #e5e7eb',
                                    borderRadius: isBlank ? '8px' : '12px',
                                    padding: '16px',
                                    boxShadow: isBlank ? 'none' : '0 1px 3px rgba(0,0,0,0.02)',
                                    display: 'flex',
                                    flexDirection: 'column'
                                }}>
                                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
                                        <div style={{ fontSize: isBlank ? '12px' : '13px', color: isBlank ? '#cbd5e1' : '#0a0a0a', fontWeight: isBlank ? '500' : '700' }}>
                                            Thread: {selectedThread.name}
                                        </div>

                                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                                            {viewMode === 'timeline' && (
                                                <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                                                    <TraceToggle
                                                        showFullTrace={showFullTrace}
                                                        onToggle={() => setShowFullTrace(!showFullTrace)}
                                                    />
                                                </div>
                                            )}

                                            <div style={{
                                                display: 'flex',
                                                gap: '4px',
                                                background: isBlank ? 'rgba(30, 41, 59, 0.5)' : '#f9fafb',
                                                padding: '3px',
                                                borderRadius: '6px',
                                                border: isBlank ? '1px solid #475569' : '1px solid #e5e7eb',
                                                height: '32px', // 统一容器高度
                                                boxSizing: 'border-box',
                                                alignItems: 'center'
                                            }}>
                                                {['timeline', 'table', 'metrics'].map(m => (
                                                    <button
                                                        key={m}
                                                        onClick={() => setViewMode(m)}
                                                        style={{
                                                            background: viewMode === m ? (isBlank ? (m === 'timeline' ? 'rgba(139, 92, 246, 0.3)' : m === 'table' ? 'rgba(16, 185, 129, 0.3)' : 'rgba(245, 158, 11, 0.3)') : '#fff') : 'transparent',
                                                            border: viewMode === m ? `1px solid ${isBlank ? (m === 'timeline' ? '#8b5cf6' : m === 'table' ? '#10b981' : '#f59e0b') : '#d1d5db'}` : '1px solid transparent',
                                                            borderRadius: '4px',
                                                            padding: '0 10px',
                                                            height: '24px', // 统一内部按钮高度
                                                            cursor: 'pointer',
                                                            color: viewMode === m ? (isBlank ? (m === 'timeline' ? '#a78bfa' : m === 'table' ? '#34d399' : '#fbbf24') : '#0a0a0a') : '#94a3b8',
                                                            fontSize: '10px',
                                                            fontWeight: '500',
                                                            transition: 'all 0.2s ease',
                                                            display: 'flex',
                                                            alignItems: 'center'
                                                        }}
                                                    >
                                                        {m.toUpperCase()}
                                                    </button>
                                                ))}
                                            </div>
                                        </div>
                                    </div>
                                    <div  style={{
                                        maxHeight: isMaximized ? '80vh' : '384px',
                                        flex: isMaximized ? 1 : 'none', // 全屏时占据剩余空间
                                        minHeight: isMaximized ? '60vh' : 'auto',

                                        overflowY: 'auto'
                                    }}>
                                        {viewMode === 'table' && <DurationTable event={durationEvent} />}
                                        {viewMode === 'timeline' && (
                                            <div style={{ background: isBlank ? 'transparent' : '#fafafa', borderRadius: '8px', padding: isBlank ? '0' : '8px' }}>
                                                <TraceVisualization events={progressEvents} onSelectTool={setSelectedEventFlow} task_id={selectedThread.id} showFullTrace={showFullTrace} selectedEventFlow={selectedEventFlow}
                                                                    tokenStreamMap={tokenStreamMap} />
                                            </div>

                                        )}

                                        {viewMode === "metrics" && (
                                            <div style={{ padding: '16px', backgroundColor: isBlank ? 'rgba(15, 23, 42, 0.8)' : '#fff', borderRadius: '8px', border: isBlank ? 'none' : '1px solid #fed7aa' }}>
                                                <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '16px' }}>
                                                    <Cpu size={14} style={{ color: '#f59e0b' }} />
                                                    <span style={{ fontSize: isBlank ? '12px' : '11px', fontWeight: isBlank ? '300' : '700', letterSpacing: isBlank ? '0.05em' : 'normal', color: isBlank ? '#fbbf24' : '#9a3412' }}>LLM STREAMING METRICS</span>
                                                </div>
                                                <MetricsStreamingPanel data={metricsEvents}/>
                                            </div>
                                        )}
                                    </div>
                                </div>
                            )}
                        </>
                    )}

                    {/* Total Views */}
                    {totalViewMode === 'table' && (
                        <div style={{ background: isBlank ? 'rgba(15, 23, 42, 0.6)' : '#f0fdf4', border: isBlank ? '1px solid #10b981' : '1px solid #bbf7d0', borderRadius: isBlank ? '8px' : '12px', padding: '16px', boxShadow: isBlank ? '0 0 20px rgba(16, 185, 129, 0.1)' : 'none' }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '16px' }}>
                                <BarChart3 size={14} style={{ color: '#10b981' }} />
                                <span style={{ fontSize: '12px', fontWeight: isBlank ? '300' : '700', letterSpacing: isBlank ? '0.05em' : 'normal', color: isBlank ? '#34d399' : '#166534' }}>TOTAL DURATION (ALL THREADS)</span>
                            </div>
                            <DurationTable event={{data: totalDurationEvent}} />
                        </div>
                    )}

                    {totalViewMode === "metrics" && (
                        <div style={{ background: isBlank ? 'rgba(15, 23, 42, 0.6)' : '#fff7ed', border: isBlank ? '1px solid #f59e0b' : '1px solid #fed7aa', borderRadius: isBlank ? '8px' : '12px', padding: isBlank ? '20px 24px' : '16px', boxShadow: isBlank ? '0 0 20px rgba(245, 158, 11, 0.1)' : 'none' }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '16px' }}>
                                <Cpu size={14} style={{ color: '#f59e0b' }} />
                                <span style={{ fontSize: '12px', fontWeight: isBlank ? '300' : '700', letterSpacing: isBlank ? '0.05em' : 'normal', color: isBlank ? '#fbbf24' : '#9a3412' }}>TOTAL LLM METRICS</span>
                            </div>
                            <MetricsStreamingPanel data={totalMetricsEvents} />
                        </div>
                    )}

                    {/* Final Summary */}
                    {finalSummary && task.status !== 'running' && (
                        <div style={{
                            marginTop: '20px', borderRadius: isBlank ? '8px' : '12px', padding: '16px', border: `1px solid ${isBlank ? '#10b981' : '#86efac'}`,
                            background: isBlank ? 'linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(20, 184, 166, 0.1))' : 'linear-gradient(135deg, #f0fdf4, #ecfdf5)'
                        }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px' }}>
                                <div style={{ width: isBlank ? '6px' : '8px', height: isBlank ? '6px' : '8px', borderRadius: '50%', backgroundColor: '#10b981' }}></div>
                                <span style={{ fontSize: '12px', fontWeight: isBlank ? '300' : '700', color: isBlank ? '#6ee7b7' : '#166534', letterSpacing: '0.5px' }}>{finalSummary.title}</span>
                            </div>
                            <pre style={{ fontSize: '12px', color: isBlank ? '#f1f5f9' : '#1f2937', whiteSpace: 'pre-wrap', lineHeight: 1.6, margin: 0, fontFamily: 'inherit', fontWeight: isBlank ? '400' : '500' }}>{finalSummary.content}</pre>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}

/**
 * @param {boolean} showFullTrace - 当前是否显示完整追踪
 * @param {function} onToggle - 切换状态的函数
 * @param {number} hiddenCount - 已隐藏（压缩）的节点数量
 * @param {boolean} isRunning - 演示是否正在进行（进行中禁用切换）
 * @param isEmbedded
 */
const TraceToggle = ({ showFullTrace, onToggle, hiddenCount=-1, isRunning , isEmbedded=true}) => {
    const { isBlank } = useGlobal();

    // --- 动态主题定义 ---
    const t = {
        bg: !isBlank ? 'rgba(255, 255, 255, 0.9)' : 'rgba(15, 23, 42, 0.9)',
        border: !isBlank ? '1px solid rgba(226, 232, 240, 1)' : '1px solid rgba(51, 65, 85, 1)',
        shadow: !isBlank ? '0 10px 25px -5px rgba(0, 0, 0, 0.08)' : '0 25px 50px -12px rgba(0, 0, 0, 0.5)',
        divider: !isBlank ? '#e2e8f0' : '#334155',
        label: !isBlank ? '#94a3b8' : '#64748b',
        btnActiveBg: !isBlank ? 'rgba(168, 85, 247, 0.1)' : 'rgba(147, 51, 234, 0.2)',
        btnActiveText: !isBlank ? '#9333ea' : '#d8b4fe',
        btnActiveBorder: !isBlank ? 'rgba(168, 85, 247, 0.3)' : 'rgba(168, 85, 247, 0.4)',
        btnIdleBg: !isBlank ? '#f8fafc' : 'rgba(30, 41, 59, 0.5)',
        btnIdleText: !isBlank ? '#64748b' : '#94a3b8',
        btnIdleBorder: !isBlank ? '#e2e8f0' : 'rgba(71, 85, 105, 0.5)',
        badgeBg: hiddenCount > 0
            ? (!isBlank ? 'rgba(239, 68, 68, 0.08)' : 'rgba(239, 68, 68, 0.1)')
            : (!isBlank ? '#f1f5f9' : '#1e293b'),
        badgeText: hiddenCount > 0 ? (!isBlank ? '#ef4444' : '#f87171') : (!isBlank ? '#94a3b8' : '#475569')
    };

    const styles = {
        container: {
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            // --- 核心修改：嵌入模式下移除外框和背景 ---
            padding: isEmbedded ? '0 8px' : '8px 16px',
            borderRadius: '8px',
            backgroundColor: isEmbedded ? 'transparent' : t.bg,
            border: isEmbedded ? 'none' : t.border,
            boxShadow: isEmbedded ? 'none' : t.shadow,
            height: isEmbedded ? '32px' : 'auto',
        },
        button: {
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            padding: '0 10px',
            height: '24px', // 与 ViewMode 按钮对齐
            borderRadius: '4px',
            cursor: isRunning ? 'not-allowed' : 'pointer',
            transition: 'all 0.2s ease',
            border: '1px solid',
            fontSize: '10px', // 缩小字号以契合 Header
            fontWeight: '600',
            textTransform: 'uppercase',
            opacity: isRunning ? 0.5 : 1,
            ...(showFullTrace ? {
                backgroundColor: t.btnActiveBg,
                color: t.btnActiveText,
                borderColor: t.btnActiveBorder,
            } : {
                backgroundColor: isBlank ? 'rgba(30, 41, 59, 0.3)' : '#f8fafc',
                color: t.btnIdleText,
                borderColor: isBlank ? '#475569' : '#e2e8f0',
            })
        },
        divider: { width: '1px', height: '14px', backgroundColor: t.divider },
        badge: {
            padding: '0 6px',
            height: '18px',
            display: 'flex',
            alignItems: 'center',
            borderRadius: '4px',
            fontSize: '9px',
            fontWeight: '800',
            backgroundColor: t.badgeBg,
            color: t.badgeText,
            border: hiddenCount > 0 ? `1px solid ${isBlank ? 'rgba(239,68,68,0.1)' : 'rgba(239,68,68,0.2)'}` : 'none',
        }
    };

    return (
        <div style={styles.container}>
            <button className="trace-toggle-btn" onClick={onToggle} disabled={isRunning} style={styles.button}>
                {showFullTrace ? <Eye size={12} /> : <EyeOff size={12} />}
                <span>{showFullTrace ? 'FULL' : 'COMPACT'}</span>
            </button>
            {/*<div style={styles.divider} />*/}
            {/*<div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>*/}
            {/*    <span style={{ fontSize: '9px', color: t.label, fontWeight: '800' }}>{hiddenCount}</span>*/}
            {/*    <div style={styles.badge}>HIDDEN</div>*/}
            {/*</div>*/}
        </div>
    );
};

function mergeLatencyFromTracesPure(traces) {
    if (!traces || typeof traces !== 'object') return [];

    const mergedMap = new Map();
    const keyOrder = [];

    // 直接遍历 Object.values(traces)，不先展开成 allData
    for (const trace of Object.values(traces)) {
        const data = trace?.durationEvent?.data;
        if (!Array.isArray(data)) continue;

        for (const item of data) {
            const { metrics, agent, tool, _call_latency } = item ?? {};

            // 严格校验必要字段
            if (
                typeof metrics !== 'string' ||
                typeof agent !== 'string' ||
                !Array.isArray(_call_latency)
            ) {
                continue;
            }

            // 使用更安全的 key（避免字段含 || 导致冲突？其实概率极低，可保留）
            const key = `${metrics}\x01${agent}\x01${tool}`; // \x01 是不可见分隔符，几乎不可能出现在字段中

            if (mergedMap.has(key)) {
                mergedMap.get(key)._call_latency.push(..._call_latency);
            } else {
                mergedMap.set(key, {
                    metrics,
                    agent,
                    tool,
                    _call_latency: [..._call_latency], // 浅拷贝 latency 数组
                });
                keyOrder.push(key);
            }
        }
    }

    return keyOrder.map(key => mergedMap.get(key));
}

