/** 认证接口 `/api/v1/auth`（契约 5.1） */

import { get, post } from './client'
import type { LoginRequest, RegisterRequest } from '@/types/dto'
import type { TokenOut, UserOut } from '@/types/models'

/** POST /auth/register —— 201 TokenOut；第一个注册的用户自动成为 admin */
export function register(payload: RegisterRequest): Promise<TokenOut> {
  return post<TokenOut>('/auth/register', payload)
}

/** POST /auth/login —— 200 TokenOut */
export function login(payload: LoginRequest): Promise<TokenOut> {
  return post<TokenOut>('/auth/login', payload)
}

/** GET /auth/me —— 200 UserOut */
export function me(): Promise<UserOut> {
  return get<UserOut>('/auth/me')
}
