import r5py

transport_network = r5py.TransportNetwork(
    "simulation/maps_data/osmnx_layers/ile-de-france-250902.osm.pbf",
    "simulation/maps_data/osmnx_layers/IDFM-gtfs.zip",  # https://prim.iledefrance-mobilites.fr/en/jeux-de-donnees/offre-horaires-tc-gtfs-idfm
)
