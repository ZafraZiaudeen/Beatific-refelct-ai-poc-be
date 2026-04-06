import { createServer } from 'node:http'
import { createApp } from './app.js'
import { env } from './config/env.js'
import { attachSessionGateway } from './sessionGateway.js'

const app = createApp()
const server = createServer(app)

attachSessionGateway(server)

server.listen(env.PORT, () => {
  console.log(`Reflect AI POC backend listening on http://localhost:${env.PORT}`)
})
