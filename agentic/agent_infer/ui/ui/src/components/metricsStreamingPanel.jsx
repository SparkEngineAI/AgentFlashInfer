// LLM指标流式面板组件
import {CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis} from "recharts";
import React from "react";
import {useGlobal} from "../context/globalContext";

const MetricsStreamingPanel = ({ data }) => {
    const { isBlank } = useGlobal();

    // 1. 获取最新的全局有效延时数据
    const latestMetrics = [...data].reverse().find(d => d.data?.metrics?.ttft_ms > 0)?.data.metrics || null;
    const latestE2ELatency = [...data].reverse().find(d => d.data?.time > 0)?.data.time || 0;

    // 2. 数据处理：聚合延迟数据 & 按 parent_id 分组数据
    // 使用 useMemo 包裹 reduce 逻辑
    const {
        ttftStats,
        tpotStats,
        p99itlStats,
        latestE2ELatencys,
        mcpData,        // 存放最终画图用的数据
        mcpKeys,        // 存放所有出现过的 Server 名字 (用于生成图例)
        agentGroups,
        latestAgentTokens
    } = React.useMemo(() => {
        return data.reduce(
            (acc, d) => {
                const metrics = d.data?.metrics;
                const time = d.data?.time;
                const serverCallArray = metrics?.mcp_server_call;
                const usage = d.data?.usage;
                const parentId = d.parent_id || 'Global';

                // 聚合延迟数据
                if (metrics?.ttft_ms > 0) acc.ttftStats.push(metrics.ttft_ms);
                if (metrics?.tpot_ms > 0) acc.tpotStats.push(metrics.tpot_ms);
                if (metrics?.p99_itl_ms > 0) acc.p99itlStats.push(metrics.p99_itl_ms);

                if (time > 0) acc.latestE2ELatencys.push(time);

                if (Array.isArray(serverCallArray) && serverCallArray.length > 0) {

                    // 1. 创建当前时间点的快照对象
                    // index 用于 X 轴，表示这是第几个数据点
                    const snapshot = { index: acc.mcpData.length };

                    // 2. 遍历数组 [3, 4]，映射成 { "Server 1": 3, "Server 2": 4 }
                    serverCallArray.forEach((count, idx) => {
                        // 生成 key，如果你知道具体名字也可以在这里做映射
                        // 比如: const names = ["Search", "Calculator"]; const key = names[idx];
                        const key = `Server ${idx + 1}`;

                        snapshot[key] = count;

                        // 记录这个 key，以便之后生成 Line 组件
                        acc.mcpKeys.add(key);
                    });

                    // 3. 推入数据数组
                    acc.mcpData.push(snapshot);
                }

                // 按 parent_id 聚合 Input Tokens
                if (usage?.input_tokens > 0) {
                    if (!acc.agentGroups[parentId]) {
                        acc.agentGroups[parentId] = [];
                    }
                    acc.agentGroups[parentId].push(usage.input_tokens);
                    acc.latestAgentTokens[parentId] = usage.input_tokens;
                }

                return acc;
            },
            { ttftStats: [], tpotStats: [], p99itlStats: [], agentGroups: {}, latestAgentTokens: {} , latestE2ELatencys: [], mcpData: [], mcpKeys: new Set()}
        );
    }, [data]); // 仅当 data 引用改变时重新计算

    const mcpServerList = Array.from(mcpKeys);

    // 3. 统计计算辅助函数
    const calculateStats = (arr) => ({
        min: arr.length > 0 ? Math.min(...arr).toFixed(2) : 0,
        max: arr.length > 0 ? Math.max(...arr).toFixed(2) : 0,
        avg: arr.length > 0 ? (arr.reduce((a, b) => a + b, 0) / arr.length).toFixed(2) : 0
    });

    // 统一的标题样式 (根据 isBlank 自动切换颜色)
    const sectionTitleStyle = {
        fontSize: '11px',
        fontWeight: '700',
        letterSpacing: '0.5px',
        textTransform: 'uppercase',
        color: isBlank ? '#94a3b8' : '#737373',
        marginBottom: '12px',
        marginTop: '8px',
        paddingLeft: '4px'
    };

    return (
        <div style={{
            height: '100%',        // 强制占据父容器 100% 高度
            width: '100%',
            display: 'flex',
            flexDirection: 'column',
            minHeight: 0,          // 关键：防止 flex 子项被内容撑开超出父级
        }}>
            {/* 滚动包装层 */}
            <div style={{
                flex: 1,           // 占据剩余空间
                overflowY: 'auto', // 只有这里产生垂直滚动
                overflowX: 'hidden',
                paddingRight: '4px', // 预留滚动条位置，防止遮挡
                display: 'flex',
                flexDirection: 'column',
                gap: '24px'
            }}>
                {/* 第一部分：顶部延时统计卡片 */}
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '16px' }}>
                    <MetricStatCard title="TTFT" unit="ms" latest={latestMetrics?.ttft_ms} stats={calculateStats(ttftStats)} color="#10b981" isBlank={isBlank} />
                    <MetricStatCard title="TPOT" unit="ms" latest={latestMetrics?.tpot_ms} stats={calculateStats(tpotStats)} color="#3b82f6" isBlank={isBlank} />
                    <MetricStatCard title="P99 ITL" unit="ms" latest={latestMetrics?.p99_itl_ms} stats={calculateStats(p99itlStats)} color="#f59e0b" isBlank={isBlank} />
                    <MetricStatCard title="E2E Lat" unit="s" latest={latestE2ELatency} stats={calculateStats(latestE2ELatencys)} color="#8b2cf6" isBlank={isBlank} />
                </div>

                {/* 第二部分：核心延时趋势图 */}
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '16px' }}>
                    <ChartCard title="TTFT Trends" data={ttftStats.map((v, i) => ({ index: i, value: v }))} dataKey="value" color="#10b981" isBlank={isBlank} />
                    <ChartCard title="TPOT Trends" data={tpotStats.map((v, i) => ({ index: i, value: v }))} dataKey="value" color="#3b82f6" isBlank={isBlank} />
                    <ChartCard title="P99 ITL Trends" data={p99itlStats.map((v, i) => ({ index: i, value: v }))} dataKey="value" color="#f59e0b" isBlank={isBlank} />
                    <ChartCard title="E2E Lat Trend" data={latestE2ELatencys.map((v, i) => ({ index: i, value: v }))} dataKey="value" color="#8b2cf6" isBlank={isBlank} />

                    {/* 👇👇👇 移动到这里的 MCP 图表 👇👇👇 */}
                    {mcpServerList.length > 0 && (
                        <MultiLineChartCard
                            // 将原本外部的标题信息整合进 Card 的 title 属性中，保持风格一致
                            title={`MCP Load (${mcpServerList.length} Active)`}
                            data={mcpData}
                            keys={mcpServerList}
                            isBlank={isBlank}
                        />
                    )}
                </div>

                {/* 分割线 */}
                <div style={{ borderTop: isBlank ? '1px solid #334155' : '1px solid #f1f5f9', margin: '8px 0' }} />

                {/* 第三部分：动态 Agent Input Tokens 区域 */}
                {agentGroups && Object.keys(agentGroups).length > 0 && (
                    <div>
                        <div style={sectionTitleStyle}>Input Tokens by Agent</div>
                        <div style={{
                            display: 'grid',
                            gridTemplateColumns: 'repeat(3, 1fr)',
                            gap: '16px'
                        }}>
                            {Object.entries(agentGroups).map(([parentId, tokens]) => (
                                <ChartCard
                                    key={parentId}
                                    // 标题显示 ID 以及最新的 Token 数，格式与卡片标题对齐
                                    title={`AGENT: ${parentId} (${latestAgentTokens[parentId] || 0} tok)`}
                                    data={tokens.map((v, i) => ({ index: i, value: v }))}
                                    dataKey="value"
                                    color="#8b5cf6"
                                    isBlank={isBlank}
                                />
                            ))}
                            {/* 无数据时的友好提示 */}
                            {Object.keys(agentGroups).length === 0 && (
                                <div style={{ gridColumn: 'span 3', padding: '40px', textAlign: 'center', color: '#94a3b8', fontSize: '12px' }}>
                                    Waiting for agent token stream...
                                </div>
                            )}
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};
// 指标统计卡片
const MetricStatCard = ({ title, unit, latest, stats, color, isBlank }) => (
    isBlank ? (
    <div style={{ background: `rgba(${hexToRgb(color)}, 0.1)`, border: `1px solid ${color}`, borderRadius: '8px', padding: '12px' }}>
            <div style={{ fontSize: '11px', color: color, fontWeight: '500', marginBottom: '8px' }}>{title}</div>
            <div style={{ fontSize: '18px', fontWeight: '300', color: '#f1f5f9', marginBottom: '8px' }}>
                {latest?.toFixed(2) || 0} <span style={{ fontSize: '11px', color: '#64748b' }}>{unit}</span>
            </div>
            <div style={{ fontSize: '10px', color: '#94a3b8', display: 'flex', justifyContent: 'space-between' }}>
                <span>Min: {stats.min} | Avg: {stats.avg} | Max: {stats.max}</span>
            </div>
        </div>
    ):(
        <div style={{ background: '#fff', border: `1.5px solid ${color}40`, borderRadius: '12px', padding: '12px', boxShadow: '0 1px 2px rgba(0,0,0,0.02)' }}>
            <div style={{ fontSize: '11px', color: color, fontWeight: '700', marginBottom: '8px', letterSpacing: '0.5px' }}>{title}</div>
            <div style={{ fontSize: '20px', fontWeight: '700', color: '#0a0a0a', marginBottom: '8px' }}>
                {latest?.toFixed(2) || 0} <span style={{ fontSize: '11px', color: '#737373', fontWeight: '500' }}>{unit}</span>
            </div>
            <div style={{ fontSize: '10px', color: '#525252', display: 'flex', justifyContent: 'space-between', fontWeight: '600' }}>
                <span>Min: {stats.min} | Avg: {stats.avg} | Max: {stats.max}</span>
            </div>
        </div>
    )
);

// 图表卡片
const ChartCard = ({ title, data, dataKey, color, isBlank }) => (
    isBlank ? (
    <div style={{ background: '#1a1f35', border: '1px solid #334155', borderRadius: '8px', padding: '12px' }}>
        <div style={{ fontSize: '11px', color: color, fontWeight: '500', marginBottom: '8px' }}>{title}</div>
        {data.length > 0 ? (
            <ResponsiveContainer width="100%" height={200}>
                <LineChart data={data} margin={{ top: 5, right: 5, left: -20, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                    <XAxis
                        dataKey="index"
                        stroke="#64748b"
                        style={{ fontSize: '10px' }}
                    />
                    <YAxis
                        stroke="#64748b"
                        style={{ fontSize: '10px' }}
                        dataKey="value"
                    />
                    <Tooltip
                        contentStyle={{ backgroundColor: '#0f172a', border: `1px solid ${color}`, borderRadius: '4px', fontSize: '11px' }}
                        cursor={{ stroke: color, strokeWidth: 1 }}
                    />
                    <Line
                        type="monotone"
                        dataKey={dataKey}
                        stroke={color}
                        dot={{ fill: color, r: 3 }}
                        activeDot={{ r: 5 }}
                        strokeWidth={2}
                        isAnimationActive={false}
                    />
                </LineChart>
            </ResponsiveContainer>
        ) : (
            <div style={{ height: '200px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#64748b', fontSize: '12px' }}>
                等待数据...
            </div>
        )}
    </div>
    ):(
        <div style={{ background: '#fff', border: '1px solid #e5e7eb', borderRadius: '12px', padding: '12px', boxShadow: '0 1px 3px rgba(0,0,0,0.02)' }}>
            <div style={{ fontSize: '11px', color: color, fontWeight: '700', marginBottom: '12px', letterSpacing: '0.5px' }}>{title}</div>
            {data.length > 0 ? (
                <ResponsiveContainer width="100%" height={200}>
                    <LineChart data={data} margin={{ top: 5, right: 5, left: -20, bottom: 5 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" /> {/* 浅色网格 */}
                        <XAxis dataKey="index" stroke="#94a3b8" style={{ fontSize: '10px', fontWeight: '500' }} />
                        <YAxis stroke="#94a3b8" style={{ fontSize: '10px', fontWeight: '500' }} dataKey="value" />
                        <Tooltip
                            contentStyle={{ backgroundColor: '#fff', border: `1px solid #e5e7eb`, borderRadius: '8px', fontSize: '11px', boxShadow: '0 4px 6px -1px rgba(0,0,0,0.1)' }}
                            itemStyle={{ fontWeight: 'bold' }}
                            cursor={{ stroke: color, strokeWidth: 1 }}
                        />
                        <Line
                            type="monotone"
                            dataKey={dataKey}
                            stroke={color}
                            dot={{ fill: color, r: 2 }}
                            activeDot={{ r: 4 }}
                            strokeWidth={2.5}
                            isAnimationActive={false}
                        />
                    </LineChart>
                </ResponsiveContainer>
            ) : (
                <div style={{ height: '200px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#a3a3a3', fontSize: '12px', fontWeight: '500' }}>
                    等待流式数据...
                </div>
            )}
        </div>
    )
);

const MultiLineChartCard = ({ title, data, keys, isBlank }) => {
    // 自动配色池
    const colors = ['#8b5cf6', '#10b981', '#3b82f6', '#f59e0b', '#ec4899', '#6366f1'];

    // 简化的样式逻辑，根据你的 isBlank 调整
    const containerStyle = isBlank
        ? { background: '#1a1f35', border: '1px solid #334155', borderRadius: '8px', padding: '12px' }
        : { background: '#fff', border: '1px solid #e5e7eb', borderRadius: '12px', padding: '12px', boxShadow: '0 1px 3px rgba(0,0,0,0.02)' };

    const titleColor = isBlank ? '#cbd5e1' : '#525252';
    const gridColor = isBlank ? '#334155' : '#f1f5f9';

    return (
        <div style={containerStyle}>
            <div style={{ fontSize: '11px', color: titleColor, fontWeight: '700', marginBottom: '12px' }}>{title}</div>
            <ResponsiveContainer width="100%" height={200}>
                <LineChart data={data} margin={{ top: 5, right: 5, left: -20, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke={gridColor} />
                    <XAxis dataKey="index" stroke="#94a3b8" style={{ fontSize: '10px' }} />
                    <YAxis stroke="#94a3b8" style={{ fontSize: '10px' }} domain={['auto', 'auto']} /> {/* 自动缩放 Y 轴 */}
                    <Tooltip
                        contentStyle={{ backgroundColor: isBlank ? '#0f172a' : '#fff', border: `1px solid ${gridColor}` }}
                        itemStyle={{ fontSize: '11px', fontWeight: 'bold' }}
                    />
                    {/* 动态生成多根线 */}
                    {keys.map((key, i) => (
                        <Line
                            key={key}
                            type="stepAfter" // 推荐：对于计数值，用阶梯图(step)通常比曲线更准确
                            dataKey={key}
                            name={key}
                            stroke={colors[i % colors.length]}
                            dot={false}
                            strokeWidth={2}
                            isAnimationActive={false} // 实时数据建议关掉动画防止闪烁
                        />
                    ))}
                </LineChart>
            </ResponsiveContainer>
        </div>
    );
};

// 颜色转RGB
const hexToRgb = (hex) => {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result ? `${parseInt(result[1], 16)}, ${parseInt(result[2], 16)}, ${parseInt(result[3], 16)}` : '0, 0, 0';
};

export default MetricsStreamingPanel;
