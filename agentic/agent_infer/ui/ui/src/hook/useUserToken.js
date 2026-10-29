// hooks/useUserToken.js
import { useState } from 'react';

export function useUserToken() {
    // 每个页面生成独立的 token，使用 sessionStorage 保证同一窗口刷新后一致，但不同窗口独立
    const [userToken] = useState(() => {
        const existing = sessionStorage.getItem('pangu_infer_user_token');
        if (existing) {
            return existing;
        }
        const newToken = 'win_' + Date.now() + '_' + Math.random().toString(36).substring(2, 8);
        sessionStorage.setItem('pangu_infer_user_token', newToken);
        return newToken;
    });

    return userToken;
}