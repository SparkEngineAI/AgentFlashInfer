import { useRef } from 'react';

/**
 * 创建一个可变的、非响应式的状态容器（类似 mini store）
 * - 初始化只执行一次
 * - 修改不会触发重渲染
 * - 永远有值，不会 null
 * - 支持 get/set/delete/has/update 等操作
 */
export function useMutableState(initialState = {}) {
    const stateRef = useRef(null);

    if (stateRef.current === null) {
        // 只初始化一次
        stateRef.current = {
            data: typeof initialState === 'function' ? initialState() : initialState,

            // 查询
            get(key) {
                return this.data[key];
            },

            // 设置（新增或修改）
            set(key, value) {
                this.data[key] = value;
            },

            // 批量设置
            setAll(obj) {
                Object.assign(this.data, obj);
            },

            // 删除
            delete(key) {
                delete this.data[key];
            },

            // 检查是否存在
            has(key) {
                return key in this.data;
            },

            // 更新（基于旧值）
            update(key, updater) {
                this.data[key] = updater(this.data[key]);
            },

            // 获取完整数据（谨慎使用，避免解构后缓存旧引用）
            getAll() {
                return this.data;
            },

            // 重置
            reset(newState = {}) {
                this.data = newState;
            }
        };
    }

    return stateRef.current;
}
