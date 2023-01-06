import os 
import redis
pool = redis.ConnectionPool(host='localhost', port=6379, decode_responses=True)
r = redis.Redis(connection_pool=pool)

PRJROOT = os.path.dirname(os.path.abspath(__file__))