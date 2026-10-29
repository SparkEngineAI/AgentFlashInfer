import assert from 'node:assert/strict';
import test from 'node:test';

import { extractFinalAnswer } from '../src/utils/formatUtil.js';

test('extracts the complete final answer from a planner result event', () => {
    const finalAnswer = '最终答案第一行。\n最终答案第二行。';
    const resultEvent = {
        data: {
            output: JSON.stringify({
                type: 'text',
                text: JSON.stringify({
                    success: true,
                    data: { final_answer: finalAnswer }
                })
            })
        }
    };

    assert.equal(extractFinalAnswer(resultEvent), finalAnswer);
});

test('ignores unsuccessful or malformed planner result events', () => {
    assert.equal(extractFinalAnswer({ data: { output: '{}' } }), null);
    assert.equal(extractFinalAnswer({ data: { output: 'not json' } }), null);
});
