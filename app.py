@st.cache_resource
def load_assets():
    try:
        import json, zipfile, tempfile, shutil, os
        import tensorflow as tf
        import h5py

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))

        for fname in ["stage1_fixed.keras", "stage2_fixed.keras", "encoder_fixed.keras", "scaler.pkl", "scaler_encoded.pkl"]:
            fpath = os.path.join(BASE_DIR, fname)
            if not os.path.exists(fpath):
                raise FileNotFoundError(f"Missing file: {fpath}")

        def fix_and_load(model_path):
            model_path = str(model_path)
            tmpdir = tempfile.mkdtemp()
            try:
                # Step 1: Read the .keras zip
                with zipfile.ZipFile(model_path, 'r') as z:
                    configs = {n: z.read(n) for n in z.namelist()}

                # Step 2: Patch config.json
                if 'config.json' in configs:
                    config = json.loads(configs['config.json'].decode('utf-8'))

                    def patch_config(obj):
                        if isinstance(obj, dict):
                            if obj.get('class_name') == 'InputLayer':
                                cfg = obj.get('config', {})
                                if 'batch_shape' in cfg:
                                    bs = cfg.pop('batch_shape')
                                    cfg['batch_input_shape'] = bs
                            for v in obj.values():
                                patch_config(v)
                        elif isinstance(obj, list):
                            for item in obj:
                                patch_config(item)
                        return obj

                    config = patch_config(config)
                    configs['config.json'] = json.dumps(config).encode('utf-8')

                # Step 3: Write patched .keras zip
                patched_path = os.path.join(tmpdir, "patched.keras")
                with zipfile.ZipFile(patched_path, 'w', zipfile.ZIP_DEFLATED) as zout:
                    for name, data in configs.items():
                        zout.writestr(name, data)

                # Step 4: Load using legacy H5 approach via model_from_json + weights
                model_config = json.loads(configs['config.json'].decode('utf-8'))
                model = tf.keras.models.model_from_json(json.dumps(model_config))

                # Step 5: Load weights from the patched keras zip
                weights_path = os.path.join(tmpdir, "weights.h5")
                with zipfile.ZipFile(patched_path, 'r') as z:
                    if 'model.weights.h5' in z.namelist():
                        with z.open('model.weights.h5') as wf:
                            with open(weights_path, 'wb') as out:
                                out.write(wf.read())
                        model.load_weights(weights_path)
                    elif 'model.h5' in z.namelist():
                        with z.open('model.h5') as wf:
                            with open(weights_path, 'wb') as out:
                                out.write(wf.read())
                        model.load_weights(weights_path)

                return model
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)

        stage1_model = fix_and_load(os.path.join(BASE_DIR, "stage1_fixed.keras"))
        stage2_model = fix_and_load(os.path.join(BASE_DIR, "stage2_fixed.keras"))
        encoder      = fix_and_load(os.path.join(BASE_DIR, "encoder_fixed.keras"))

        scaler         = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
        scaler_encoded = joblib.load(os.path.join(BASE_DIR, "scaler_encoded.pkl"))

        return stage1_model, stage2_model, encoder, scaler, scaler_encoded

    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()


assets = load_assets()
if assets is None:
    st.error("Model loading failed. Check logs.")
    st.stop()
stage1_model, stage2_model, encoder, scaler, scaler_encoded = assets
