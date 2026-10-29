import React from 'react';

const ScrollableContent = ({ content, isRunning, agentColor, isBlank, isResultCard, cardIsDeleted }) => (
    <div style={{
        maxHeight: isResultCard ? '320px' : '180px',
        overflowY: 'auto',
        padding: '10px 12px',
        borderRadius: '6px',
        background: isBlank ? 'rgba(15, 23, 42, 0.55)' : '#f8fafc',
        color: isBlank ? '#cbd5e1' : '#334155',
        borderLeft: `3px solid ${agentColor || '#06b6d4'}`,
        opacity: cardIsDeleted ? 0.55 : 1,
        whiteSpace: 'pre-wrap',
        wordBreak: 'break-word',
        fontSize: '12px',
        lineHeight: 1.6,
    }}>
        {content == null ? '' : typeof content === 'string' ? content : JSON.stringify(content, null, 2)}
        {isRunning && <span aria-label="running"> ▍</span>}
    </div>
);

export default ScrollableContent;
