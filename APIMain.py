from conduit.app import create_app
from conduit.settings import ProdConfig, DevConfig, runtime_settings

THREADING, PORT, DEBUG = runtime_settings()

# if __name__ == '__main__':
app = create_app(ProdConfig)
app.run(threaded=True, port=5000, debug=True)
