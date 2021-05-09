from conduit import create_app
from conduit.settings import DevConfig, runtime_settings

PORT, DEBUG = runtime_settings()
app = create_app(DevConfig)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT)
