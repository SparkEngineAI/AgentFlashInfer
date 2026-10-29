import React, { useState, useEffect } from 'react';
import { X, Settings, Save, RotateCcw, Check } from 'lucide-react';
import { useGlobal } from '../context/globalContext';

const ConfigModal = ({ isOpen, onClose, userToken, config, onConfigChange }) => {
    const [localConfig, setLocalConfig] = useState({});
    const [saving, setSaving] = useState(false);
    const [saved, setSaved] = useState(false);
    const [hasLoadedConfig, setHasLoadedConfig] = useState(false);

    // 配置项定义 - 分为三块
    const configItems = [
        // ===== 工具优化配置 =====
        { key: 'mcp_hub_switch', label: 'MCP Hub 负载均衡', description: '启用 MCP 服务器的负载均衡功能', category: '工具优化', type: 'switch', default: true },
        { key: 'tool_predict_switch', label: '工具预测', description: '使用工具预测模型来筛选候选工具', category: '工具优化', type: 'switch', default: true },
        { key: 'tool_predict_url', label: '工具预测 URL', description: '工具预测模型的 API 地址', category: '工具优化', type: 'input', default: '', showWhen: { tool_predict_switch: true } },
        { key: 'tool_predict_appcode', label: '工具预测 AppCode', description: '工具预测服务应用代码', category: '工具优化', type: 'input', default: '', showWhen: { tool_predict_switch: true } },

        // ===== 推理优化配置 =====
        { key: 'agent_infer_switch', label: 'AgentInfer 加速', description: '启用 AgentInfer 加速功能（开启后自动启用以下所有子开关）', category: '推理优化', type: 'switch', default: false },
        { key: 'resum_compress_async', label: '异步压缩', description: '启用异步压缩以减少 token 使用', category: '推理优化', type: 'switch', default: true, showWhen: { agent_infer_switch: true } },
        { key: 'early_stop_enabled', label: '提前停止', description: '在适当的时候提前停止搜索', category: '推理优化', type: 'switch', default: true, showWhen: { agent_infer_switch: true } },
        { key: 'model_routing_switch', label: '模型路由', description: '在大模型和小模型间自动切换', category: '推理优化', type: 'switch', default: true, showWhen: { agent_infer_switch: true } },

        // AgentInfer 开启时允许配置的参数
        { key: 'early_stop_min_iter', label: '提前停止最小迭代', description: '提前停止的最小迭代次数', category: '推理优化', type: 'slider', min: 5, max: 30, step: 1, default: 10, showWhen: { agent_infer_switch: true } },
        { key: 'SEARCH_COMPRESS_URL', label: '搜索压缩 URL', description: '搜索结果压缩模型的 API 地址', category: '推理优化', type: 'input', default: '', showWhen: { agent_infer_switch: true } },
        { key: 'SEARCH_COMPRESS_API_KEY', label: '搜索压缩 API Key', description: '搜索结果压缩模型的 API 密钥', category: '推理优化', type: 'input', default: '', showWhen: { agent_infer_switch: true } },
        { key: 'SEARCH_COMPRESS_MODEL_NAME', label: '搜索压缩模型名称', description: '搜索结果压缩模型的名称', category: '推理优化', type: 'input', default: 'pangu_auto', showWhen: { agent_infer_switch: true } },
        { key: 'LARGE_MODEL_URL', label: '大模型 URL', description: '大模型的 API 地址', category: '推理优化', type: 'input', default: '', showWhen: { agent_infer_switch: true } },
        { key: 'SMALL_MODEL_URL', label: '小模型 URL', description: '小模型的 API 地址', category: '推理优化', type: 'input', default: '', showWhen: { agent_infer_switch: true } },
        { key: 'routing_model_name', label: '路由模型名称', description: '模型路由使用的模型名称', category: '推理优化', type: 'input', default: 'pangu_auto', showWhen: { agent_infer_switch: true } },

        // ===== 参数配置 =====
        // 基础模型配置
        { key: 'BASE_URL', label: 'Base URL', description: 'LLM 服务的基础 URL (OPENAI_BASE_URL)', category: '参数配置', type: 'input', default: '' },
        { key: 'API_KEY', label: 'API Key', description: 'LLM 服务的 API 密钥 (OPENAI_API_KEY)', category: '参数配置', type: 'input', default: '' },
        { key: 'MODEL_NAME', label: '模型名称', description: '要使用的模型名称 (EXAMPLE_MODEL_NAME)', category: '参数配置', type: 'input', default: '' },

        // Agent 配置
        {
            key: 'PLANNER_MODE',
            label: 'Planner 模式',
            description: '选择 Planner 的工作模式',
            category: '参数配置',
            type: 'select',
            options: [
                { value: 'auto', label: 'Auto (自动)' },
                { value: 'writing', label: 'Writing (写作)' },
                { value: 'qa', label: 'QA (问答)' }
            ],
            default: 'auto'
        },

        // 其他参数配置
        { key: 'model_temperature', label: '模型温度', description: '模型的生成温度 (0-1)', category: '参数配置', type: 'slider', min: 0, max: 1, step: 0.1, default: 0.3 },
        { key: 'model_max_tokens', label: '最大 Token 数', description: '响应的最大 token 数', category: '参数配置', type: 'slider', min: 1024, max: 16000, step: 1024, default: 5120 },
        { key: 'TOKEN_THRESHOLD', label: 'Token 阈值', description: '触发任务完成的 token 阈值', category: '参数配置', type: 'slider', min: 2048, max: 100000, step: 2048, default: 40960 },
        { key: 'PLANNER_MAX_TURNS', label: 'Planner 最大轮次', description: 'Planner Agent 的最大轮次数', category: '参数配置', type: 'slider', min: 5, max: 50, step: 1, default: 10 },
        { key: 'SEEKER_MAX_TURNS', label: 'Seeker 最大轮次', description: 'Seeker Agent 的最大轮次数', category: '参数配置', type: 'slider', min: 5, max: 50, step: 1, default: 15 },
        { key: 'WRITER_MAX_TURNS', label: 'Writer 最大轮次', description: 'Writer Agent 的最大轮次数', category: '参数配置', type: 'slider', min: 5, max: 50, step: 1, default: 20 },

        // 记忆功能
        { key: 'ENABLE_MEMORY', label: '启用记忆', description: '启用长期记忆功能', category: '参数配置', type: 'switch', default: false },
        { key: 'ENABLE_SHARED_MEMORY', label: '启用共享记忆', description: '启用跨会话共享记忆', category: '参数配置', type: 'switch', default: false },
    ];

    // 检查配置项是否应该显示
    const shouldShowItem = (item) => {
        if (!item.showWhen) return true;
        for (const [key, value] of Object.entries(item.showWhen)) {
            if (localConfig[key] !== value) return false;
        }
        return true;
    };

    // 当 Modal 关闭时重置状态，当 Modal 打开时加载配置
    useEffect(() => {
        if (isOpen) {
            setHasLoadedConfig(false);
        } else {
            setLocalConfig({});
            setHasLoadedConfig(false);
        }
    }, [isOpen]);

    // 从后端加载配置
    useEffect(() => {
        if (isOpen && userToken && !hasLoadedConfig) {
            fetch(`/api/window/config/${userToken}`)
                .then(res => res.json())
                .then(data => {
                    const fullConfig = {};
                    const isDefault = data._is_default !== false;

                    configItems.forEach(item => {
                        if (item.type === 'switch') {
                            fullConfig[item.key] = data[item.key] ?? item.default ?? false;
                        } else if (item.type === 'select') {
                            fullConfig[item.key] = data[item.key] ?? item.options[0].value;
                        } else if (item.type === 'slider') {
                            fullConfig[item.key] = data[item.key] ?? item.default;
                        } else if (item.default !== undefined) {
                            fullConfig[item.key] = data[item.key] ?? item.default;
                        } else {
                            fullConfig[item.key] = data[item.key] ?? '';
                        }
                    });

                    // 同步 agent_infer_switch 子开关状态
                    if (fullConfig.agent_infer_switch === false) {
                        fullConfig.resum_compress_async = false;
                        fullConfig.early_stop_enabled = false;
                        fullConfig.model_routing_switch = false;
                    }

                    fullConfig._is_default = isDefault;
                    setLocalConfig(fullConfig);
                    onConfigChange(fullConfig);
                    setHasLoadedConfig(true);
                })
                .catch(err => {
                    console.error('Failed to load config:', err);
                    const defaults = {};
                    configItems.forEach(item => {
                        if (item.type === 'switch') {
                            defaults[item.key] = item.default ?? false;
                        } else if (item.type === 'select') {
                            defaults[item.key] = item.options[0].value;
                        } else if (item.type === 'slider') {
                            defaults[item.key] = item.default;
                        } else if (item.default !== undefined) {
                            defaults[item.key] = item.default;
                        } else {
                            defaults[item.key] = '';
                        }
                    });
                    setLocalConfig(defaults);
                    setHasLoadedConfig(true);
                });
        }
    }, [isOpen, userToken, hasLoadedConfig]);

    // agent_infer_switch 变化时的处理
    const handleAgentInferToggle = (newValue) => {
        const updates = {
            agent_infer_switch: newValue
        };

        // 当开启 agent_infer_switch 时，自动设置子开关为 true
        // 当关闭 agent_infer_switch 时，自动设置子开关为 false
        updates.resum_compress_async = newValue;
        updates.early_stop_enabled = newValue;
        updates.model_routing_switch = newValue;

        setLocalConfig(prev => ({
            ...prev,
            ...updates
        }));
    };

    const handleToggle = (key) => {
        if (key === 'agent_infer_switch') {
            handleAgentInferToggle(!localConfig[key]);
        } else {
            setLocalConfig(prev => ({
                ...prev,
                [key]: !prev[key]
            }));
        }
    };

    const handleInputChange = (key, value) => {
        setLocalConfig(prev => ({
            ...prev,
            [key]: value
        }));
    };

    const handleSelectChange = (key, value) => {
        setLocalConfig(prev => ({
            ...prev,
            [key]: value
        }));
    };

    const handleSliderChange = (key, value) => {
        setLocalConfig(prev => ({
            ...prev,
            [key]: parseFloat(value)
        }));
    };

    const handleSave = async () => {
        setSaving(true);
        setSaved(false);
        try {
            const fullConfig = {};
            configItems.forEach(item => {
                if (item.type === 'switch') {
                    fullConfig[item.key] = localConfig[item.key] ?? item.default ?? false;
                } else if (item.type === 'select') {
                    fullConfig[item.key] = localConfig[item.key] ?? item.options[0].value;
                } else if (item.type === 'slider') {
                    fullConfig[item.key] = localConfig[item.key] ?? item.default;
                } else if (item.default !== undefined) {
                    fullConfig[item.key] = localConfig[item.key] ?? item.default;
                } else {
                    fullConfig[item.key] = localConfig[item.key] ?? '';
                }
            });

            // 如果 agent_infer_switch 关闭，强制将子开关设置为 false
            if (fullConfig.agent_infer_switch === false) {
                fullConfig.resum_compress_async = false;
                fullConfig.early_stop_enabled = false;
                fullConfig.model_routing_switch = false;
            }

            const response = await fetch('/api/window/config', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    user_token: userToken,
                    config_overrides: fullConfig
                }),
            });

            if (response.ok) {
                setSaved(true);
                fullConfig._is_default = false;
                onConfigChange(fullConfig);
                setTimeout(() => {
                    onClose();
                }, 500);
            } else {
                console.error('Failed to save config:', await response.text());
            }
        } catch (error) {
            console.error('Error saving config:', error);
        } finally {
            setSaving(false);
        }
    };

    const handleReset = () => {
        const defaults = {};
        configItems.forEach(item => {
            if (item.type === 'switch') {
                defaults[item.key] = item.default ?? false;
            } else if (item.type === 'select') {
                defaults[item.key] = item.options[0].value;
            } else if (item.type === 'slider') {
                defaults[item.key] = item.default;
            } else if (item.default !== undefined) {
                defaults[item.key] = item.default;
            } else {
                defaults[item.key] = '';
            }
        });
        setLocalConfig(defaults);
    };

    if (!isOpen) return null;

    const { isBlank } = useGlobal();

    const theme = {
        overlay: 'rgba(0, 0, 0, 0.5)',
        bg: isBlank ? '#0f172a' : '#ffffff',
        border: isBlank ? '#334155' : '#e5e7eb',
        textPrimary: isBlank ? '#f1f5f9' : '#0a0a0a',
        textSecondary: isBlank ? '#64748b' : '#525252',
        inputBg: isBlank ? '#1e293b' : '#f9fafb',
        accent: isBlank ? '#10b981' : '#06b6d4',
        accentHover: isBlank ? '#059669' : '#0891b2',
        success: isBlank ? '#10b981' : '#22c55e',
        categoryBg: isBlank ? '#1e293b' : '#f1f5f9',
        categoryText: isBlank ? '#cbd5e1' : '#475569',
        scrollbarThumb: isBlank ? '#334155' : '#d1d5db',
    };

    const categories = ['工具优化', '推理优化', '参数配置'];

    return (
        <div
            style={{
                position: 'fixed',
                top: 0,
                left: 0,
                right: 0,
                bottom: 0,
                backgroundColor: theme.overlay,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                zIndex: 1000,
            }}
            onClick={onClose}
        >
            <div
                style={{
                    backgroundColor: theme.bg,
                    border: `1px solid ${theme.border}`,
                    borderRadius: '12px',
                    width: '90%',
                    maxWidth: '750px',
                    maxHeight: '85vh',
                    display: 'flex',
                    flexDirection: 'column',
                    boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.25)',
                }}
                onClick={(e) => e.stopPropagation()}
            >
                {/* Header */}
                <div
                    style={{
                        padding: '20px 24px',
                        borderBottom: `1px solid ${theme.border}`,
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                    }}
                >
                    <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                        <Settings size={20} style={{ color: theme.accent }} />
                        <span
                            style={{
                                fontSize: '16px',
                                fontWeight: '700',
                                color: theme.textPrimary,
                            }}
                        >
                            窗口配置
                        </span>
                        <span
                            style={{
                                fontSize: '12px',
                                color: theme.textSecondary,
                                fontWeight: '500',
                            }}
                        >
                            Token: {userToken}
                        </span>
                    </div>
                    <button
                        onClick={onClose}
                        style={{
                            background: 'none',
                            border: 'none',
                            color: theme.textSecondary,
                            cursor: 'pointer',
                            padding: '4px',
                            display: 'flex',
                        }}
                    >
                        <X size={20} />
                    </button>
                </div>

                {/* Content */}
                <div
                    style={{
                        flex: 1,
                        overflow: 'auto',
                        padding: '20px 24px',
                    }}
                >
                    {categories.map((category) => {
                        const categoryItems = configItems
                            .filter((item) => item.category === category)
                            .filter((item) => shouldShowItem(item));

                        if (categoryItems.length === 0) return null;

                        return (
                            <div key={category} style={{ marginBottom: '24px' }}>
                                <div
                                    style={{
                                        fontSize: '12px',
                                        fontWeight: '700',
                                        color: theme.categoryText,
                                        marginBottom: '12px',
                                        letterSpacing: '0.05em',
                                        textTransform: 'uppercase',
                                    }}
                                >
                                    {category}
                                </div>
                                <div
                                    style={{
                                        backgroundColor: theme.categoryBg,
                                        borderRadius: isBlank ? '8px' : '10px',
                                        padding: '16px',
                                        border: `1px solid ${theme.border}`,
                                    }}
                                >
                                    {categoryItems.map((item, idx) => (
                                        <div
                                            key={item.key}
                                            style={{
                                                padding: '10px 0',
                                                borderBottom:
                                                    idx < categoryItems.length - 1
                                                        ? `1px solid ${theme.border}`
                                                        : 'none',
                                            }}
                                        >
                                            <div
                                                style={{
                                                    display: 'flex',
                                                    justifyContent: 'space-between',
                                                    alignItems: 'center',
                                                }}
                                            >
                                                <div style={{ flex: 1 }}>
                                                    <div
                                                        style={{
                                                            fontSize: '13px',
                                                            fontWeight: '600',
                                                            color: theme.textPrimary,
                                                            marginBottom: '2px',
                                                        }}
                                                    >
                                                        {item.label}
                                                    </div>
                                                    <div
                                                        style={{
                                                            fontSize: '11px',
                                                            color: theme.textSecondary,
                                                        }}
                                                    >
                                                        {item.description}
                                                    </div>
                                                </div>
                                                {item.type === 'switch' && (
                                                    <button
                                                        onClick={() => handleToggle(item.key)}
                                                        style={{
                                                            position: 'relative',
                                                            width: '44px',
                                                            height: '24px',
                                                            backgroundColor:
                                                                localConfig[item.key]
                                                                    ? theme.accent
                                                                    : isBlank
                                                                    ? '#334155'
                                                                    : '#e5e7eb',
                                                            border: 'none',
                                                            borderRadius: '12px',
                                                            cursor: 'pointer',
                                                            transition: 'background-color 0.2s',
                                                            flexShrink: 0,
                                                            marginLeft: '16px',
                                                        }}
                                                    >
                                                        <span
                                                            style={{
                                                                position: 'absolute',
                                                                top: '2px',
                                                                left: localConfig[item.key]
                                                                    ? '22px'
                                                                    : '2px',
                                                                width: '20px',
                                                                height: '20px',
                                                                backgroundColor: '#fff',
                                                                borderRadius: '50%',
                                                                transition: 'left 0.2s',
                                                                boxShadow: '0 2px 4px rgba(0,0,0,0.2)',
                                                            }}
                                                        />
                                                    </button>
                                                )}
                                                {item.type === 'input' && (
                                                    <input
                                                        type="text"
                                                        value={localConfig[item.key] ?? ''}
                                                        onChange={(e) => handleInputChange(item.key, e.target.value)}
                                                        placeholder={item.default ?? ''}
                                                        style={{
                                                            marginLeft: '16px',
                                                            minWidth: '200px',
                                                            maxWidth: '300px',
                                                            padding: '6px 10px',
                                                            borderRadius: '6px',
                                                            border: `1px solid ${theme.border}`,
                                                            backgroundColor: theme.inputBg,
                                                            color: theme.textPrimary,
                                                            fontSize: '12px',
                                                            outline: 'none',
                                                        }}
                                                    />
                                                )}
                                                {item.type === 'select' && (
                                                    <select
                                                        value={localConfig[item.key] || item.options[0].value}
                                                        onChange={(e) =>
                                                            handleSelectChange(item.key, e.target.value)
                                                        }
                                                        style={{
                                                            padding: '6px 12px',
                                                            borderRadius: '6px',
                                                            border: `1px solid ${theme.border}`,
                                                            backgroundColor: theme.inputBg,
                                                            color: theme.textPrimary,
                                                            fontSize: '12px',
                                                            marginLeft: '16px',
                                                            cursor: 'pointer',
                                                        }}
                                                    >
                                                        {item.options.map((option) => (
                                                            <option key={option.value} value={option.value}>
                                                                {option.label}
                                                            </option>
                                                        ))}
                                                    </select>
                                                )}
                                                {item.type === 'slider' && (
                                                    <div
                                                        style={{
                                                            marginLeft: '16px',
                                                            minWidth: '140px',
                                                            display: 'flex',
                                                            alignItems: 'center',
                                                            gap: '8px',
                                                        }}
                                                    >
                                                        <input
                                                            type="range"
                                                            min={item.min}
                                                            max={item.max}
                                                            step={item.step}
                                                            value={localConfig[item.key] ?? item.default}
                                                            onChange={(e) =>
                                                                handleSliderChange(item.key, e.target.value)
                                                            }
                                                            style={{ flex: 1, cursor: 'pointer' }}
                                                        />
                                                        <span
                                                            style={{
                                                                fontSize: '12px',
                                                                color: theme.textSecondary,
                                                                minWidth: '35px',
                                                                textAlign: 'right',
                                                            }}
                                                        >
                                                            {localConfig[item.key] ?? item.default}
                                                        </span>
                                                    </div>
                                                )}
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        );
                    })}
                </div>

                {/* Footer */}
                <div
                    style={{
                        padding: '16px 24px',
                        borderTop: `1px solid ${theme.border}`,
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                    }}
                >
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <button
                            onClick={handleReset}
                            style={{
                                display: 'flex',
                                alignItems: 'center',
                                gap: '6px',
                                padding: '8px 14px',
                                borderRadius: '6px',
                                border: `1px solid ${theme.border}`,
                                backgroundColor: 'transparent',
                                color: theme.textSecondary,
                                cursor: 'pointer',
                                fontSize: '12px',
                                fontWeight: '600',
                            }}
                        >
                            <RotateCcw size={14} />
                            重置
                        </button>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        {saved && (
                            <span
                                style={{
                                    display: 'flex',
                                    alignItems: 'center',
                                    gap: '4px',
                                    fontSize: '12px',
                                    color: theme.success,
                                    fontWeight: '500',
                                }}
                            >
                                <Check size={14} />
                                已保存
                            </span>
                        )}
                        <button
                            onClick={handleSave}
                            disabled={saving}
                            style={{
                                display: 'flex',
                                alignItems: 'center',
                                gap: '6px',
                                padding: '8px 16px',
                                borderRadius: '6px',
                                border: 'none',
                                backgroundColor: saving
                                    ? (isBlank ? '#334155' : '#d1d5db')
                                    : theme.accent,
                                color: '#fff',
                                cursor: saving ? 'not-allowed' : 'pointer',
                                fontSize: '12px',
                                fontWeight: '600',
                                opacity: saving ? 0.7 : 1,
                            }}
                        >
                            <Save size={14} />
                            {saving ? '保存中...' : '保存配置'}
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default ConfigModal;