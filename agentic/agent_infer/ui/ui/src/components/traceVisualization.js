// TraceVisualization Component
import {use, useCallback, useEffect, useMemo, useReducer, useRef, useState} from "react";
import {AlertCircle, ChevronDown, ChevronUp, Loader2, Zap} from "lucide-react";
import {useMutableState} from "../hook/useMutableState";
import {parseIfJson} from "../utils/formatUtil";
import {useGlobal} from "../context/globalContext";
import ScrollableContent from "./scrollableContent";

const TraceVisualization = ({ events={}, onSelectTool, task_id, showFullTrace=false, selectedEventFlow, tokenStreamMap}) => {

    const traceStatesRef = useRef(new Map());
    // 获取或初始化当前 task_id 的状态
    if (!traceStatesRef.current.has(task_id)) {
        traceStatesRef.current.set(task_id, {
            agentGroups: {},
            processedEventIds: new Set(),
            spanMap: new Map(),
            toolsMap: new Map(),
            childAgentKeys: new Set(),
        });
    }
    const currentState = traceStatesRef.current.get(task_id);
    const getToolIcon = () => {
        return  '🔧';
    };
    const getGenerationIcon = (modelRouting) => {
        // 注入动画样式（仅执行一次）
        if (typeof document !== 'undefined' && !document.getElementById('model-icon-animations')) {
            const style = document.createElement('style');
            style.id = 'model-icon-animations';
            style.textContent = `
            @keyframes largeModelPulse {
                0%, 100% { 
                    transform: scale(1);
                    box-shadow: 0 0 8px rgba(236, 72, 153, 0.3), inset 0 0 4px rgba(236, 72, 153, 0.1);
                }
                50% { 
                    transform: scale(1.05);
                    box-shadow: 0 0 14px rgba(236, 72, 153, 0.5), inset 0 0 8px rgba(236, 72, 153, 0.15);
                }
            }
            @keyframes largeModelInnerGlow {
                0%, 100% { 
                    opacity: 0.3;
                    transform: scale(1);
                }
                50% { 
                    opacity: 0.5;
                    transform: scale(0.95);
                }
            }
            @keyframes largeModelOuterGlow {
                0%, 100% { 
                    transform: scale(1);
                    opacity: 0.3;
                }
                50% { 
                    transform: scale(1.08);
                    opacity: 0.6;
                }
            }
            @keyframes largeModelSpin {
                0% { transform: rotate(0deg) scale(1); }
                25% { transform: rotate(90deg) scale(1.05); }
                50% { transform: rotate(180deg) scale(1); }
                75% { transform: rotate(270deg) scale(1.05); }
                100% { transform: rotate(360deg) scale(1); }
            }
        `;
            document.head.appendChild(style);
        }

        if(modelRouting && modelRouting === "large_model") {
            return (
                <div style={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    width: '1em',
                    height: '1em',
                    borderRadius: '4px',
                    background: 'linear-gradient(135deg, rgba(236, 72, 153, 0.15), rgba(168, 85, 247, 0.15))',
                    border: '1px solid rgba(236, 72, 153, 0.5)',
                    position: 'relative',
                    animation: 'largeModelPulse 2s ease-in-out infinite',
                    cursor: 'pointer',
                    transition: 'all 0.3s ease',
                    boxSizing: 'border-box',
                    fontSize: 'inherit',
                    marginRight: '0.55em',
                }}>
                    {/* 内层闪光 */}
                    <div style={{
                        position: 'absolute',
                        top: '1px',
                        left: '1px',
                        right: '1px',
                        bottom: '1px',
                        borderRadius: '3px',
                        background: 'linear-gradient(135deg, rgba(236, 72, 153, 0.08), rgba(168, 85, 247, 0.08))',
                        animation: 'largeModelInnerGlow 1.5s ease-in-out infinite',
                        pointerEvents: 'none',
                    }} />

                    {/* 外层光晕 */}
                    <div style={{
                        position: 'absolute',
                        top: '-2px',
                        left: '-2px',
                        right: '-2px',
                        bottom: '-2px',
                        borderRadius: '5px',
                        border: '0.5px solid rgba(236, 72, 153, 0.3)',
                        animation: 'largeModelOuterGlow 2.5s ease-in-out infinite',
                        opacity: 0.5,
                        pointerEvents: 'none',
                    }} />

                    {/* 核心图标 - 闪电符号 */}
                    <span style={{
                        position: 'relative',
                        zIndex: 1,
                        fontSize: '1em',
                        animation: 'largeModelSpin 3s linear infinite',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        lineHeight: 1,
                    }}>
                    ⚡
                </span>
                </div>
            );
        }

        return '💭';
    };

    // 获取智能体头像
    const getAgentAvatar = (agentName) => {
        return '🧠';
    };

    // 获取智能体颜色
    const getAgentColor = (agentName) => {
        const colors = {
            'PlannerAgent': '#10b981',
            'ObjectiveInformationSeekerAgent': '#3b82f6',
            'WriterAgent': '#f59e0b',
            'ResearchAgent': '#8b5cf6',
            'AnalyzerAgent': '#ef4444'
        };
        return colors[agentName] || '#6b7280';
    };

    // 获取智能体背景
    const getAgentDarkBg = (agentName) => {
        const bgs = {
            'PlannerAgent': 'rgba(16, 185, 129, 0.1)',
            'ObjectiveInformationSeekerAgent': 'rgba(59, 130, 246, 0.1)',
            'WriterAgent': 'rgba(245, 158, 11, 0.1)',
            'ResearchAgent': 'rgba(139, 92, 246, 0.1)',
            'AnalyzerAgent': 'rgba(239, 68, 68, 0.1)'
        };
        return bgs[agentName] || 'rgba(107, 114, 128, 0.1)';
    };

    function processSingleEvent(event, agents, spanMap, toolsMap) {
        const { event: eventType, span_type, data = {}, span_id, parent_id } = event;
        spanMap.set(span_id, event);

        if (event.event === 'on_trace_start' || event.event === 'on_trace_end') {
            return false;
        }

        let changed = false;

        const findParentAgent = (parentId) => {
            let parent = spanMap.get(parentId);
            if (!parent) return null;
            while (parent && parent.span_type !== 'agent') {
                parent = spanMap.get(parent.parent_id);
            }
            return parent?.span_type === 'agent' ? parent : null;
        };

        switch (span_type) {
            case 'mcp_tools':
                if (eventType === 'on_span_end' && data.available_tools?.length > 0) {
                    const parentAgent = findParentAgent(parent_id);
                    if (parentAgent) {
                        const agentName = parentAgent.data.agent_name || 'Unknown Agent';
                        const agentKey = `${agentName}_${parentAgent.span_id}`;
                        if (agents[agentKey] && agents[agentKey].availableTools.length === 0) {
                            agents[agentKey].availableTools = data.available_tools;
                            changed = true;
                        }
                    }
                }
                break;
            case 'agent':
                if (eventType === 'on_span_start') {
                    const agentName = data.agent_name || 'Unknown Agent';
                    const agentKey = `${agentName}_${span_id}`;
                    if (!agents[agentKey]) {
                        agents[agentKey] = {
                            agentName,
                            avatar: getAgentAvatar(agentName),
                            color: getAgentColor(agentName),
                            darkBg: getAgentDarkBg(agentName),
                            tools: [],
                            availableTools: [],
                            _spanId: span_id,
                            generation: [],
                            last_generation: null,
                            actions: [],
                            actionsLen: 0,
                            compressionStates:[]
                        };
                        changed = true;
                    }
                    const parentSpan = spanMap.get(parent_id);
                    if (parentSpan && parentSpan.span_type === 'function') {
                        const parentTool = toolsMap.get(parent_id);
                        if (parentTool) {
                            if (!parentTool.childrenAgents) parentTool.childrenAgents = [];
                            parentTool.childrenAgents.push(agents[agentKey]);
                            currentState.childAgentKeys.add(agentKey);
                            changed = true;
                        }
                    }
                }
                break;
            case 'function':
                if (eventType === 'on_span_start') {
                    const parentAgent = findParentAgent(parent_id);
                    if (parentAgent) {
                        const agentName = parentAgent.data.agent_name || 'Unknown Agent';
                        const agentKey = `${agentName}_${parentAgent.span_id}`;
                        const toolObj = {
                            id: span_id,
                            name: data.name,
                            status: 'running',
                            fullInput: JSON.stringify(data.input || {}, null, 2),
                            displayedInput: '',
                            input: JSON.stringify(data.input || {}),
                            output: '',
                            icon: getToolIcon(data.name),
                            childrenAgents: [],
                            index: agents[agentKey].actionsLen
                        };
                        toolsMap.set(span_id, toolObj);
                        if (agents[agentKey]) {
                            agents[agentKey].tools.push(toolObj);
                            agents[agentKey].actions.push({ type: 'tool', data: toolObj });
                            agents[agentKey].actionsLen = agents[agentKey].actionsLen+1
                            changed = true;
                        }
                    }
                } else if (eventType === 'on_span_end') {
                    const tool = toolsMap.get(span_id);
                    if (tool) {
                        tool.input = parseIfJson(data.input || '""')
                        tool.status = 'completed';
                        tool.output = parseIfJson(data.output || '""')
                        tool.displayedInput = tool.fullInput; // 补全
                        changed = true;
                    }
                    const parentAgent = findParentAgent(parent_id);
                    if (parentAgent) {
                        const agentName = parentAgent.data.agent_name || 'Unknown Agent';
                        const agentKey = `${agentName}_${parentAgent.span_id}`;
                        if (agents[agentKey]) {
                            const idx = agents[agentKey].tools.findIndex(t => t.id === span_id);
                            if (idx !== -1) {
                                agents[agentKey].tools[idx].status = 'completed';
                                agents[agentKey].tools[idx].output = parseIfJson(data.output || '""');
                                changed = true;
                            }
                        }
                        // console.log(`🔵 [Action 添加] ${span_type} -> ${agentKey}`, { span_id });

                    }
                }
                break;
            case 'generation':
                if (eventType === 'on_span_start') {
                    const parentAgent = findParentAgent(parent_id);
                    if (parentAgent) {
                        const agentName = parentAgent.data.agent_name || 'Unknown Agent';
                        const agentKey = `${agentName}_${parentAgent.span_id}`;
                        const genObj = {
                            id: span_id,
                            name: data.name,
                            status: 'running',
                            fullInput: parseIfJson(data.messages || ""),
                            displayedInput: '',
                            icon: getGenerationIcon(data?.model_config?.current_model ?? ''),
                            messages: [],
                            response: [],
                            index: agents[agentKey].actionsLen
                        };
                        if (agents[agentKey]) {
                            agents[agentKey].generation.push(genObj);
                            agents[agentKey].actions.push({ type: 'generation', data: genObj });
                            agents[agentKey].actionsLen = agents[agentKey].actionsLen+1
                            changed = true;
                        }
                    }
                } else if (eventType === 'on_span_end') {
                    const parentAgent = findParentAgent(parent_id);
                    if (parentAgent) {
                        const agentName = parentAgent.data.agent_name || 'Unknown Agent';
                        const agentKey = `${agentName}_${parentAgent.span_id}`;
                        if (agents[agentKey]) {
                            const idx = agents[agentKey].generation.findIndex(g => g.id === span_id);
                            if (idx !== -1) {
                                agents[agentKey].generation[idx].status = 'completed';
                                agents[agentKey].generation[idx].messages = parseIfJson(data.messages || []);
                                agents[agentKey].generation[idx].response = parseIfJson(data.response || []);
                                changed = true;
                            }
                        }
                    }
                }
                break;
            case 'custom':
                if(data?.name === 'generation_resum_compress' && eventType === 'on_span_start'){
                    const parentAgent = findParentAgent(parent_id);
                    if (parentAgent && parentAgent.data && parentAgent.span_id) {
                        const agentName = parentAgent.data.agent_name || 'Unknown Agent';
                        const agentKey = `${agentName}_${parentAgent.span_id}`;
                        const existing = agents[agentKey]?.compressionState || {};
                        const already_compressed_num = data.already_compressed_num || 0
                        agents[agentKey].compressionState={
                                isCompressing: true,
                                compressingFromIdx: ( data.compress_till_idx_not_inc || 0 ) + already_compressed_num,
                                already_compressed_num: already_compressed_num,
                                compressedIndices: new Set(
                                    existing.compressedIndices ? [...existing.compressedIndices] : []
                                ),
                                summaryMap: new Map(existing.summaryMap || []),

                        };
                    }
                }else if (data?.name === 'generation_resum_compress' && eventType === 'on_span_end'){
                    const parentAgent = findParentAgent(parent_id);
                    if (parentAgent && parentAgent.data && parentAgent.span_id) {
                        const agentName = parentAgent.data.agent_name || 'Unknown Agent';
                        const agentKey = `${agentName}_${parentAgent.span_id}`;
                        const existing = agents[agentKey]?.compressionState || {};
                        const already_compressed_num = existing.already_compressed_num || 0

                        const compressedIndices = new Set(Array.from(data.excluded_indices, idx => idx + already_compressed_num));
                        const summaryMap = existing.summaryMap;
                        if (data.last_think_idx != null && data.final_summary != null) {
                            summaryMap.set(data.last_think_idx+already_compressed_num,data.final_summary);
                        }
                        agents[agentKey].compressionState={
                            ...agents[agentKey].compressionState,
                            isCompressing: false,
                            compressedIndices: new Set([...existing.compressedIndices, ...compressedIndices]),
                            summaryMap: existing.summaryMap,
                        };
                    }
                }
                break;
            default:
                let event = event;
        }

        return changed;
    }
    // ====== Render-time incremental build ======
    if (events) {
        let hasChange = false;
        for (const event of events) {
            const { span_id, event: eventType } = event;
            const eventId = `${span_id}_${eventType}`;

            if (!currentState.processedEventIds.has(eventId)) {
                currentState.processedEventIds.add(eventId);
                const changed = processSingleEvent(
                    event,
                    currentState.agentGroups,
                    currentState.spanMap,
                    currentState.toolsMap
                );
                if (changed) hasChange = true;
            }
        }
        // 注意：不需要 forceUpdate！ref 已更新，本轮 render 即生效
    }
    const filteredAgents = {};
    for (const [key, agent] of Object.entries(currentState.agentGroups)) {
        if (!currentState.childAgentKeys.has(key)) {
            filteredAgents[key] = agent;
        }
    }



    const containerRef = useRef(null); // 指向带滚动条的那个 div
    const isInitialRender = useRef(true); // 记录是否是第一次加载

    useEffect(() => {
        const container = containerRef.current;
        if (!container) return;

        const { scrollTop, scrollHeight, clientHeight } = container;

        // --- 核心判断逻辑 ---
        // threshold 是阈值（单位像素），比如 50px。
        // 如果用户向上滚动超过了 50px，我们就认为他正在阅读历史，不强行拉回。
        const threshold = 50;
        const isAtBottom = scrollHeight - scrollTop - clientHeight <= threshold;

        // 如果是第一次渲染，或者用户本来就在底部，则自动滚动
        if (isInitialRender.current || isAtBottom) {
            container.scrollTo({
                top: scrollHeight,
                behavior: 'smooth'
            });
            isInitialRender.current = false;
        }
    }, [events]); // 只有当事件数组增加时触发

    const allProcessedIds = new Set();
    Object.values(currentState.agentGroups).forEach(agent => {
        agent.actions.forEach(action => {
            if (allProcessedIds.has(action.data.id)) {
                console.error(`🚨 [致命冲突] 发现跨 Agent 的重复 Action ID: ${action.data.id}`);
            }
            allProcessedIds.add(action.data.id);
        });
    });

    return (
        <div
            ref={containerRef}
            style={{
                display: 'flex',
                flexDirection: 'column',
                gap: '16px',
            }}
        >
            {Object.keys(filteredAgents).length === 0 ? (
                <div style={{ textAlign: 'center', color: '#64748b', padding: '20px', fontSize: '13px' }}>
                    ⏳ 等待执行中...
                </div>
            ) : (
                Object.entries(filteredAgents).map(([agentKey, agentData]) => (
                    <AgentCard
                        key={agentKey}
                        agentData={currentState.agentGroups[agentKey]}
                        onSelectTool={onSelectTool}
                        showFullTrace={showFullTrace}
                        selectedEventFlow={selectedEventFlow}
                        tokenStreamMap={tokenStreamMap}
                    />
                ))
            )}
        </div>
    );
}

// 👇 AgentCard 组件 —— 支持展开/收起全部内容（包括 tools 和 generation）
const AgentCard = ({  agentData, onSelectTool, showFullTrace=false, selectedEventFlow,tokenStreamMap }) => {
    const hasGeneration = Array.isArray(agentData?.generation) && agentData?.generation?.length > 0;
    const hasTools = Array.isArray(agentData?.tools) && agentData?.tools?.length > 0;
    const hasExpandableContent = hasGeneration || hasTools;
    const MAX_EXPANDED = 1; // 👈 新增：最多同时展开 2 个输入/消息区域
    const compressionState = agentData?.compressionState;
    const [showContent, setShowContent] = useState(true); // 默认展开（可按需改为 false）
    const [expandedInputs, setExpandedInputs] = useState(new Set());
    const toggleInput = useCallback((id, shouldExpand) => {
        setExpandedInputs(prev => {
            const next = new Set(prev);
            if (shouldExpand) {
                if (next.size >= MAX_EXPANDED) {
                    const first = next.values().next().value;
                    next.delete(first);
                }
                next.add(id);
            } else {
                next.delete(id);
            }
            return next;
        });
    }, []);

    return (
        <div
            style={{
                background: agentData.darkBg,
                border: `1px solid ${agentData.color}`,
                borderRadius: '8px',
                padding: '10px',
                boxShadow: `0 0 15px ${agentData.color}20`
            }}
            onClick={(e) => {
                e.stopPropagation();
            }}
        >
            {/* Agent Header —— 可点击展开/收起 */}
            <div
                style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px',
                    cursor: hasExpandableContent ? 'pointer' : 'default'
                }}
                onClick={() => {
                    if (hasExpandableContent) {
                        setShowContent(!showContent);
                    }
                }}
            >
                <span style={{ fontSize: '18px' }}>{agentData.avatar}</span>
                <span style={{ color: agentData.color, fontWeight: '500', fontSize: '13px' }}>
                    {agentData.agentName}
                                    <span style={{ opacity: 0.5, fontSize: '11px', marginLeft: '4px' }}>
                        ({agentData._spanId})
                    </span>
                </span>
                {hasExpandableContent && (
                    <button
                        onClick={(e) => {
                            e.stopPropagation();
                            setShowContent(!showContent);
                        }}
                        style={{
                            marginLeft: 'auto',
                            background: 'none',
                            border: 'none',
                            color: agentData.color,
                            cursor: 'pointer',
                            fontSize: '10px',
                            padding: 0,
                            lineHeight: 1
                        }}
                    >
                        {showContent ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
                    </button>
                )}
            </div>

            {/* 内容区域：仅在 showContent 为 true 时渲染 */}
            {showContent && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0px' }}>
                    {(agentData.actions || []).map((action, idx) => {
                        if (action.type === 'tool') {
                            return (
                                <ToolCard
                                    key={action.data.id}
                                    tool={action.data}
                                    agentColor={agentData.color}
                                    onSelectTool={onSelectTool}
                                    isInputExpanded={expandedInputs.has(action.data.id)}
                                    onToggleInput={(expand) => toggleInput(action.data.id, expand)}
                                    compressionState={compressionState}
                                    showFullTrace={showFullTrace}
                                    selectedEventFlow={selectedEventFlow}
                                    tokenStreamMap={tokenStreamMap}
                                />
                            );
                        } else if (action.type === 'generation') {
                            return (
                                <GenerationCard
                                    key={action.data.id}
                                    generation={action.data}
                                    agentColor={agentData.color}
                                    onSelectTool={onSelectTool}
                                    isMessagesExpanded={expandedInputs.has(action.data.id)}
                                   onToggleMessages={(expand) => toggleInput(action.data.id, expand)}
                                    // onToggleMessages={toggleInput}
                                    compressionState={compressionState}
                                    showFullTrace={showFullTrace}
                                    selectedEventFlow={selectedEventFlow}
                                    tokenStreamMap={tokenStreamMap}
                                />
                            );
                        }
                        return null;
                    })}
                </div>
            )}
        </div>
    );
};
const ToolCard = ({ tool, agentColor, onSelectTool, isInputExpanded, onToggleInput, compressionState,showFullTrace=false, selectedEventFlow, tokenStreamMap }) => {
    const { isBlank } = useGlobal();
    const hasChildrenAgents = Array.isArray(tool.childrenAgents) && tool.childrenAgents.length > 0;
    useEffect(() => {
        if (tool.status === 'completed') onToggleInput(true);
    }, [tool.status]);
    const isCompressed = !!compressionState?.compressedIndices?.has?.(tool.index);
    const isInCompressionRange =
        compressionState?.isCompressing &&
        tool.index <= compressionState?.compressingFromIdx;
    const isRunning =  tool.status === 'running';
    return (
        <CompressedBaseCard
            key={tool.id}
            isBlank={isBlank}
            agentColor={agentColor}
            status={tool.status}
            icon={tool.icon}
            title={tool.name}
            onClick={() => onSelectTool(tool)}
            isExpanded={isInputExpanded}
            content={tool.output}
            cardIndex={tool.index}
            compressionState={compressionState}
            showFullTrace={showFullTrace}
            isSelected={selectedEventFlow && selectedEventFlow?.id === tool.id}
            contentElement={
                <ScrollableContent
                    content={tool.output}
                    isRunning={isRunning}
                    agentColor={agentColor}
                    isBlank={isBlank}
                    cardIsDeleted={compressionState?.compressedIndices?.has(tool.index)}
                />
            }
        >
            {hasChildrenAgents && (
                <div style={{
                    marginTop: isBlank ? '8px' : '10px',
                    paddingLeft: isBlank ? '12px' : '14px',
                    borderLeft: isBlank ? `1px dashed ${agentColor}40` : `2px solid ${agentColor}20`,
                    display: 'flex',
                    flexDirection: 'column',
                    gap: isBlank ? '0' : '8px'
                }}>
                    {tool.childrenAgents.map((childAgent, idx) => (
                        <AgentCard
                            key={`${childAgent.agentName}_${childAgent._spanId || idx}`}
                            agentData={childAgent}
                            onSelectTool={onSelectTool}
                            showFullTrace={showFullTrace}
                            selectedEventFlow={selectedEventFlow}
                            tokenStreamMap={tokenStreamMap}
                        />
                    ))}
                </div>
            )}
        </CompressedBaseCard>
    );
};
const GenerationCard = ({generation, agentColor, onSelectTool, isMessagesExpanded, onToggleMessages, compressionState,showFullTrace=false,selectedEventFlow,tokenStreamMap }) => {
    const { isBlank } = useGlobal();
    // 自动展开逻辑
    useEffect(() => {
        // 只要状态是 running（流式开始）或者 completed（已完成），就触发展开
        // 如果当前已经是展开状态 (!isMessagesExpanded)，则不需要重复触发
        if (
            (generation.status === 'running' || generation.status === 'completed') &&
            !isMessagesExpanded
        ) {
            onToggleMessages(true);
        }
    }, [generation.status]); // 监听 status 的变化
    let displayContent = tokenStreamMap?.[generation.id]
        || generation.response?.[0]?.content
        || "";
    if(compressionState?.summaryMap && compressionState.summaryMap?.has( generation.index ) ){
        displayContent = compressionState.summaryMap.get(generation.index);
    }

    const isRunning = generation.status === 'running';
    return (
        <CompressedBaseCard
            key={generation.id}
            isBlank={isBlank}
            agentColor={agentColor}
            status={generation.status}
            icon={generation.icon}
            title="generation"
            onClick={() => onSelectTool(generation)}
            isExpanded={isMessagesExpanded}
            content={displayContent}
            cardIndex={generation.index}
            compressionState={compressionState}
            isSelected={selectedEventFlow && selectedEventFlow?.id === generation.id}
            contentElement={
                <ScrollableContent
                    content={displayContent}
                    isRunning={isRunning}
                    agentColor={agentColor}
                    isBlank={isBlank}
                    isResultCard={compressionState?.summaryMap?.has(generation.index)}
                    cardIsDeleted={compressionState?.compressedIndices?.has(generation.index)}
                />
            }
        />
    );
};

const CompressedBaseCard = ({
                                isBlank,
                                agentColor,
                                status,
                                icon,
                                title,
                                onClick,
                                children,
                                isExpanded,
                                content,
                                cardIndex,
                                compressionState = {},
                                showFullTrace=false,
                                isSelected = false, // 由外部控制
                                contentElement
                            }) => {
    const [isHovered, setIsHovered] = useState(false);
    const [isDeleting, setIsDeleting] = useState(false);

    useEffect(() => {
        if (typeof document === 'undefined' || document.getElementById('basecard-compression-v3')) return;
        const style = document.createElement('style');
        style.id = 'basecard-compression-v3';
        style.textContent = `
      @keyframes blink { 50% { opacity: 0; } }
      
      @keyframes progress-shimmer {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
      }

      @keyframes loader-pulse {
        0%, 100% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.02); opacity: 0.8; }
      }
      
      @keyframes spin-icon {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
      }

      @keyframes greenSummaryCrystalGlow {
        0%, 100% { 
            box-shadow: 0 0 15px rgba(16, 185, 129, 0.4), inset 0 0 10px rgba(16, 185, 129, 0.1);
            border-color: rgba(16, 185, 129, 0.6);
        }
        50% { 
            box-shadow: 0 0 25px rgba(16, 185, 129, 0.7), inset 0 0 15px rgba(16, 185, 129, 0.2);
            border-color: rgba(16, 185, 129, 1);
        }
      }

      @keyframes deleteSlideOut {
        0% { opacity: 1; transform: translateY(0) scale(1); filter: blur(0px); }
        30% { opacity: 0.8; transform: translateY(-20px) scale(0.98); }
        100% { opacity: 0; transform: translateY(-100px) translateX(-50px) scale(0.5); filter: blur(8px); }
      }

      @keyframes collapseHeight {
        0% { margin-bottom: 4px; }
        100% { height: 0; margin: 0; padding: 0; visibility: hidden; }
      }
      
       .icon-spinner {
        animation: spin-icon 1s linear infinite !important;
        display: inline-block;
        margin-right: 5px;
      }
    `;
        document.head.appendChild(style);
    }, []);

    let {
        isCompressing = false,
        compressingFromIdx = null,
        compressedIndices = new Set(),
        summaryMap = new Map(),
        already_compressed_num = -1
    } = compressionState;

    const isResultCard = summaryMap.has(cardIndex);
    let displayIsExpanded = isExpanded;
    if (isResultCard) {
        displayIsExpanded = true;
    }
    const isInCompressionRange = compressingFromIdx !== null && cardIndex <= compressingFromIdx;
    const cardIsCompressing = isInCompressionRange && isCompressing;
    const cardIsDeleted = !showFullTrace && compressedIndices.has(cardIndex);

    const relativeWeight = compressingFromIdx !== 0 ? (cardIndex / compressingFromIdx) : 0;
    const layerOpacity = 0.4 - (relativeWeight * 0.3);
    const midIndex = compressingFromIdx !== null ? Math.floor(compressingFromIdx / 2) : 0;
    const isMidPoint = cardIndex === midIndex;

    useEffect(() => {
        if (cardIsDeleted && !isResultCard && !showFullTrace) {
            setIsDeleting(true);
        } else {
            setIsDeleting(false);
        }
    }, [cardIsDeleted, isResultCard, showFullTrace]);

    const isRunning = status === 'running';
    const displayIcon = isRunning ? <Loader2 size={14} className="icon-spinner" style={{ color: agentColor }} /> : icon;

    const getCardStyle = () => {
        let baseStyle = {
            transition: 'all 0.2s ease',
            marginBottom: '4px',
            position: 'relative',
            cursor: 'pointer',
            overflow: 'hidden',
        };

        // 核心改动：选中时的边框厚度和颜色
        const borderWidth = isSelected ? '2px' : '1px';
        const borderColor = (isHovered || isSelected) ? agentColor : `${agentColor}40`;

        if (isBlank) {
            baseStyle = {
                ...baseStyle,
                backgroundColor: cardIsDeleted ? '#f1f5f9' : (isHovered ? `${agentColor}15` : '#0f172a'),
                border: `${borderWidth} solid ${cardIsDeleted ? '#cbd5e1' : borderColor}`,
                borderRadius: '4px',
                padding: '8px 10px',
                fontSize: '11px',
                color: cardIsDeleted ? '#94a3b8' : '#cbd5e1',
            };
        } else {
            baseStyle = {
                ...baseStyle,
                backgroundColor: cardIsDeleted ? '#f1f5f9' : (isHovered ? `${agentColor}08` : '#ffffff'),
                border: `${borderWidth} solid ${cardIsDeleted ? '#cbd5e1' : borderColor}`,
                borderRadius: '8px',
                padding: '10px 12px',
                fontSize: '12px',
                color: cardIsDeleted ? '#94a3b8' : '#404040',
            };
        }

        // 摘要卡片的特殊动画（最高优先级）
        if (isResultCard) {
            baseStyle.animation = 'greenSummaryCrystalGlow 2s ease-in-out infinite';
            baseStyle.border = '2px solid #10b981';
            baseStyle.backgroundColor = 'rgba(16, 185, 129, 0.08)';
            baseStyle.zIndex = 5;
        } else if (isDeleting) {
            baseStyle.animation = 'deleteSlideOut 0.8s ease-in forwards, collapseHeight 0.5s ease-in 0.3s forwards';
            baseStyle.pointerEvents = 'auto';
        }

        // 选中时稍微提升层级，确保边框不被遮挡
        if (isSelected) baseStyle.zIndex = 2;

        return baseStyle;
    };

    return (
        <div
            onClick={(e) => {
                if (onClick) onClick(e);
            }}
            onMouseEnter={() => setIsHovered(true)}
            onMouseLeave={() => setIsHovered(false)}
            style={getCardStyle()}
        >
            {cardIsCompressing && (
                /* ... 压缩动画保持不变 ... */
                <div style={{
                    position: 'absolute',
                    inset: 0,
                    zIndex: 20,
                    pointerEvents: 'auto',
                    backdropFilter: 'blur(1.5px)',
                    background: `linear-gradient(135deg, rgba(16, 185, 129, ${layerOpacity + 0.15}) 0%, rgba(16, 185, 129, ${layerOpacity}) 100%)`,
                    border: `1px solid rgba(16, 185, 129, ${layerOpacity + 0.3})`,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center'
                }}>
                    {isMidPoint && (
                        <div style={{
                            width: '85%',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '10px',
                            animation: 'loader-pulse 2s infinite ease-in-out'
                        }}>
                            <div style={{
                                flex: 1,
                                height: '6px',
                                backgroundColor: 'rgba(255,255,255,0.3)',
                                borderRadius: '10px',
                                overflow: 'hidden',
                                position: 'relative',
                                border: '1px solid rgba(16, 185, 129, 0.4)',
                                boxShadow: '0 0 12px rgba(16, 185, 129, 0.5)'
                            }}>
                                <div style={{
                                    position: 'absolute',
                                    top: 0,
                                    left: 0,
                                    width: '100%',
                                    height: '100%',
                                    background: 'linear-gradient(90deg, #10b981, #6ee7b7, #10b981, #6ee7b7, #10b981)',
                                    backgroundSize: '200% 100%',
                                    animation: 'progress-shimmer 1s infinite linear'
                                }} />
                            </div>
                            <div style={{
                                fontSize: '10px',
                                fontWeight: '900',
                                color: '#ffffff',
                                backgroundColor: '#10b981',
                                padding: '2px 6px',
                                borderRadius: '4px',
                                textShadow: '0 1px 2px rgba(0,0,0,0.2)',
                                letterSpacing: '0.05em',
                                whiteSpace: 'nowrap',
                                boxShadow: '0 2px 4px rgba(16, 185, 129, 0.4)'
                            }}>
                                COMPRESSING
                            </div>
                        </div>
                    )}
                </div>
            )}

            <div style={{ position: 'absolute', top: '6px', right: '8px', zIndex: 10 }}>
                {isResultCard && (
                    <div style={{
                        padding: '2px 6px',
                        background: 'rgba(16, 185, 129, 0.2)',
                        color: '#059669',
                        fontSize: '9px',
                        fontWeight: '700',
                        borderRadius: '3px',
                        border: '1px solid rgba(16, 185, 129, 0.5)'
                    }}>
                        💎 摘要
                    </div>
                )}
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <span style={{
                    fontFamily: 'monospace',
                    fontSize: '10px',
                    color: isResultCard ? '#10b981' : (cardIsDeleted ? '#94a3b8' : agentColor),
                    opacity: isSelected ? 1 : 0.6, // 选中时序号变亮
                    backgroundColor: isBlank ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.03)',
                    padding: '0 4px',
                    borderRadius: '2px'
                }}>
                    #{String(cardIndex).padStart(2, '0')}
                </span>

                <strong style={{
                    color: isResultCard ? '#059669' : (cardIsDeleted ? '#94a3b8' : agentColor),
                    fontWeight: isBlank ? 'bold' : (isSelected ? '800' : '700'), // 选中时标题略微加粗
                    textTransform: isBlank ? 'none' : 'uppercase',
                    textDecoration: cardIsDeleted && !isResultCard ? 'line-through' : 'none',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '4px'
                }}>
                    {isResultCard ? `✨ ${title}` : <>{displayIcon} {title}</>}
                </strong>
            </div>

            {displayIsExpanded && contentElement}

            {children}
        </div>
    );
};

export default TraceVisualization;