import React, { createContext, useContext, useState } from 'react';

const GlobalContext = createContext(undefined);

export function GlobalProvider({ children }) {
    const [isBlank, setIsBlank] = useState(true);

    return (
        <GlobalContext.Provider value={{ isBlank, setIsBlank }}>
            {children}
        </GlobalContext.Provider>
    );
}

export function useGlobal() {
    const context = useContext(GlobalContext);
    if (!context) {
        throw new Error('useGlobal must be used within GlobalProvider');
    }
    return context;
}
