// .umirc.ts
// @ts-ignore
import { defineConfig } from 'umi';
console.log('✅ .umirc.ts 被成功加载！');

export default defineConfig({
    // 基础配置
    base: '/',
    publicPath: '/',
    routes: [
        { path: '/', component: 'index' },
        // { path: '/about', component: 'about' },
    ],
    proxy: {
        '/api': {
            target: 'http://localhost:8289',
            changeOrigin: true,
            secure: false,
            onProxyReq: (proxyReq, req, res) => {
                console.log('[PROXY] 请求转发:', req.method, req.url);
            },
            onProxyRes: (proxyRes, req, res) => {
                console.log('[PROXY] 响应收到:', proxyRes.statusCode, req.url);
            },
        }
    }
    // 启用必要的插件
    // layout: {},
    // 如果用了 antd
    // antd: {},
});