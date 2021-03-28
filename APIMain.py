from conduit.app import create_app
from conduit.settings import ProdConfig, DevConfig, runtime_settings

THREADING, PORT, DEBUG = runtime_settings()

if __name__ == '__main__':
    app = create_app(DevConfig)
    app.run(host='0.0.0.0', port=5000)  # (threaded=True, port=5000, debug=True)
