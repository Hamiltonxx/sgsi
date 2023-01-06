from application import app
# import redis
# pool = redis.ConnectionPool(host='0.0.0.0', port=10001, decode_responses=True)
# r = redis.Redis(connection_pool=pool)

app.debug = True 

#app.run(host="0.0.0.0",port=10000, )
app.run(host="0.0.0.0", port="10000", ssl_context=('yijian.pem','yijian.key'))
