from conduit.app import create_app
from conduit.settings import ProdConfig, DevConfig, runtime_settings

THREADING, PORT, DEBUG = runtime_settings()
app = create_app(DevConfig)

if __name__ == '__main__':

    app.run()  # (threaded=True, port=5000, debug=True)
