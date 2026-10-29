import React, { useEffect } from 'react';
import { GlobalProvider } from '../context/globalContext';
import { useUserToken } from '../hook/useUserToken';
import MultiAgentPlatform from './platformPage';

const IndexPage = () => {
    const userToken = useUserToken();

    useEffect(() => {
        const notifyClose = () => {
            const body = new Blob([JSON.stringify({ user_token: userToken })], {
                type: 'application/json',
            });
            navigator.sendBeacon('/api/user/close', body);
        };
        window.addEventListener('beforeunload', notifyClose);
        return () => window.removeEventListener('beforeunload', notifyClose);
    }, [userToken]);

    return (
        <GlobalProvider>
            <MultiAgentPlatform userToken={userToken} />
        </GlobalProvider>
    );
};

export default IndexPage;
