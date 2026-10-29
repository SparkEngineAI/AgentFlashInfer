export function formatString(value) {
    if (typeof value !== 'string') return value;
    return value.replace(/\\n/g, '\n').replace(/\\"/g, '"');
}

export function parseIfJson(value) {
    if (typeof value !== 'string') return value;
    try {
        return JSON.parse(value);
    } catch {
        return formatString(value);
    }
}

export function extractFinalAnswer(resultEvent) {
    let payload = parseIfJson(resultEvent?.data?.output);
    if (payload && typeof payload === 'object' && 'text' in payload) {
        payload = parseIfJson(payload.text);
    }

    const finalAnswer = payload?.success === true
        ? payload?.data?.final_answer
        : null;
    return typeof finalAnswer === 'string' && finalAnswer.trim()
        ? finalAnswer
        : null;
}
