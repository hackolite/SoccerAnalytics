# Stubs (cache files)

Pickle stub files are generated automatically on the first run and cached here
to speed up subsequent runs:

- `track_stubs.pkl` — object tracking results
- `camera_movement_stub.pkl` — camera movement estimation results

To force re-generation, delete the relevant `.pkl` file or set
`read_from_stub=False` in `main.py`.
