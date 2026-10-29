// MetricsTable Component
import {useGlobal} from "../context/globalContext";

const DurationTable = ({ event }) => {
    // 从事件流中提取状态更新数据
    const statusUpdateEvent = event;
    let metrics = [];
    const { isBlank, setIsBlank } = useGlobal();

    if (statusUpdateEvent && statusUpdateEvent.data) {
        metrics = statusUpdateEvent.data.map(item => {
            // 计算count和total
            const count = item._call_latency.length;
            const total = item._call_latency.reduce((sum, val) => sum + val, 0);
            const avg = count > 0 ? total / count : 0;

            return {
                metrics: item.metrics,
                agent: item.agent,
                tool: item.tool || '-',
                count,
                total: parseFloat(total.toFixed(3)),
                avg: parseFloat(avg.toFixed(3))
            };
        });
    }

    const totalLatency = metrics.reduce((sum, m) => sum + m.total, 0);

    return isBlank ? (
        <div style={{ overflowX: 'auto', border: '1px solid #334155', borderRadius: '6px', backgroundColor: '#0f172a' }}>
            <table style={{ width: '100%', fontSize: '11px', fontFamily: 'monospace', borderCollapse: 'collapse' }}>
                <thead>
                <tr style={{ borderBottom: '1px solid #334155', backgroundColor: '#1e293b' }}>
                    <th style={{ textAlign: 'left', padding: '8px 10px', color: '#94a3b8' }}>METRICS</th>
                    <th style={{ textAlign: 'left', padding: '8px 10px', color: '#94a3b8' }}>AGENT</th>
                    <th style={{ textAlign: 'left', padding: '8px 10px', color: '#94a3b8' }}>TOOL</th>
                    <th style={{ textAlign: 'right', padding: '8px 10px', color: '#94a3b8' }}>AVG LATENCY</th>
                    <th style={{ textAlign: 'right', padding: '8px 10px', color: '#94a3b8' }}>COUNT</th>
                    <th style={{ textAlign: 'right', padding: '8px 10px', color: '#94a3b8' }}>TOTAL</th>
                    <th style={{ textAlign: 'right', padding: '8px 10px', color: '#94a3b8' }}>PROPORTION</th>
                </tr>
                </thead>
                <tbody>
                {metrics.length === 0 ? (
                    <tr>
                        <td colSpan="7" style={{ textAlign: 'center', padding: '12px', color: '#64748b' }}>No metrics yet</td>
                    </tr>
                ) : (
                    metrics.map((metric, idx) => (
                        <tr key={idx} style={{ borderBottom: '1px solid #1e293b', backgroundColor: idx % 2 === 0 ? 'rgba(30,41,59,0.3)' : 'transparent' }}>
                            <td style={{ padding: '8px 10px', color: '#cbd5e1' }}>
                                {metric.metrics}
                            </td>
                            <td style={{ padding: '8px 10px', color: '#f1f5f9' }}>{metric.agent}</td>
                            <td style={{ padding: '8px 10px', color: '#f1f5f9' }}>{metric.tool}</td>
                            <td style={{ padding: '8px 10px', textAlign: 'right', color: '#10b981' }}>{metric.avg.toFixed(3)}</td>
                            <td style={{ padding: '8px 10px', textAlign: 'right', color: '#cbd5e1' }}>{metric.count}</td>
                            <td style={{ padding: '8px 10px', textAlign: 'right', color: '#cbd5e1' }}>{metric.total.toFixed(3)}</td>
                            <td style={{ padding: '8px 10px', textAlign: 'right', color: '#64748b' }}>
                                {totalLatency > 0 ? ((metric.total / totalLatency) * 100).toFixed(1) : 0}%
                            </td>
                        </tr>
                    ))
                )}
                </tbody>
            </table>
        </div>
    ):(
        <div style={{
            overflowX: 'auto',
            border: '1px solid #e5e7eb',
            borderRadius: '12px',
            backgroundColor: '#fff',
            boxShadow: '0 1px 3px rgba(0,0,0,0.02)'
        }}>
            <table style={{ width: '100%', fontSize: '12px', fontFamily: 'inherit', borderCollapse: 'collapse' }}>
                <thead>
                <tr style={{
                    borderBottom: '2px solid #f3f4f6',
                    backgroundColor: '#f9fafb'
                }}>
                    <th style={{ textAlign: 'left', padding: '12px 10px', color: '#525252', fontWeight: '700', letterSpacing: '0.5px' }}>METRICS</th>
                    <th style={{ textAlign: 'left', padding: '12px 10px', color: '#525252', fontWeight: '700', letterSpacing: '0.5px' }}>AGENT</th>
                    <th style={{ textAlign: 'left', padding: '12px 10px', color: '#525252', fontWeight: '700', letterSpacing: '0.5px' }}>TOOL</th>
                    <th style={{ textAlign: 'right', padding: '12px 10px', color: '#525252', fontWeight: '700', letterSpacing: '0.5px' }}>AVG LATENCY</th>
                    <th style={{ textAlign: 'right', padding: '12px 10px', color: '#525252', fontWeight: '700', letterSpacing: '0.5px' }}>COUNT</th>
                    <th style={{ textAlign: 'right', padding: '12px 10px', color: '#525252', fontWeight: '700', letterSpacing: '0.5px' }}>TOTAL</th>
                    <th style={{ textAlign: 'right', padding: '12px 10px', color: '#525252', fontWeight: '700', letterSpacing: '0.5px' }}>PROPORTION</th>
                </tr>
                </thead>
                <tbody>
                {metrics.length === 0 ? (
                    <tr>
                        <td colSpan="7" style={{ textAlign: 'center', padding: '24px', color: '#a3a3a3', fontWeight: '500' }}>No metrics yet</td>
                    </tr>
                ) : (
                    metrics.map((metric, idx) => (
                        <tr key={idx} style={{
                            borderBottom: '1px solid #f3f4f6',
                            backgroundColor: idx % 2 === 0 ? '#ffffff' : '#fafafa',
                            transition: 'background-color 0.2s'
                        }}>
                            <td style={{ padding: '10px', color: '#06b6d4', fontWeight: '700', fontFamily: 'monospace' }}>
                                {metric.metrics}
                            </td>
                            <td style={{ padding: '10px', color: '#0a0a0a', fontWeight: '600' }}>{metric.agent}</td>
                            <td style={{ padding: '10px', color: '#404040', fontWeight: '500' }}>{metric.tool}</td>
                            <td style={{ padding: '10px', textAlign: 'right', color: '#059669', fontWeight: '700', fontFamily: 'monospace' }}>
                                {metric.avg.toFixed(3)}ms
                            </td>
                            <td style={{ padding: '10px', textAlign: 'right', color: '#0a0a0a', fontWeight: '500' }}>{metric.count}</td>
                            <td style={{ padding: '10px', textAlign: 'right', color: '#0a0a0a', fontWeight: '600', fontFamily: 'monospace' }}>{metric.total.toFixed(3)}</td>
                            <td style={{ padding: '10px', textAlign: 'right', color: '#737373', fontWeight: '600' }}>
                                <div style={{ display: 'inline-block', width: '45px' }}>
                                    {totalLatency > 0 ? ((metric.total / totalLatency) * 100).toFixed(1) : 0}%
                                </div>
                            </td>
                        </tr>
                    ))
                )}
                </tbody>
            </table>
        </div>
    );
};

export default DurationTable;
