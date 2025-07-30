from simulation.preparing_resources import IleDeFranceMobilityDataCollector

# log = Logger("data/logs")
# log.log_it("apple.com.csv")

# IleDeFranceMobilityDataCollector().ile_de_france_open_street_map()

# from pyrosm import OSM
# import matplotlib.pyplot as plt
# osm = OSM("simulation/data/chunk_4.osm.pbf")
# network = osm.get_network("driving")
# plt.figure(dpi=300)
# ax = network.plot(figsize=(10, 10), linewidth=0.5, edgecolor="gray")
# plt.title("Pyrosm Network Plot")
# plt.show()
# del network
# osm = OSM("simulation/data/chunk_4.osm.pbf")
# buildings = osm.get_network()
# buildings.plot()
# print(buildings)

# preprocess_maps.py
import osmnx as ox
# G_ = ox.io.load_graphml("./simulation/data/osmnx_layers/IDF_network.graphml")
# G_.nodes()
# G_
# # Plot the graph
# fig, ax = ox.plot_graph(G_)

# You can customize the plot with various parameters:
# fig, ax = ox.plot_graph(G_, node_color='r', node_size=10, edge_linewidth=0.5, edge_color='gray', bgcolor='w')
# print("Starting one-time map pre-processing...")
place_name = "Île-de-France, France"
#
# --- Process Drivable Roads ---
print("Extracting drivable network...")
G_drive = ox.graph_from_place(place_name, network_type='drive', simplify=True, retain_all=True)
ox.save_graphml(G_drive, filepath='./simulation/data/osmnx_layers/IDF_drive_network.graphml')
print("Saved drivable network to disk.")

# --- Process Walkable Paths ---
print("Extracting walkable network...")
G_walk = ox.graph_from_place(place_name, network_type='walk', simplify=True, retain_all=True)
ox.save_graphml(G_walk, filepath='./simulation/data/osmnx_layers/IDF_walk_network.graphml')
print("Saved walkable network to disk.")


print("Extracting walkable network...")
G_bike = ox.graph_from_place(place_name, network_type='bike', simplify=True, retain_all=True)
ox.save_graphml(G_bike, filepath='./simulation/data/osmnx_layers/IDF_bike_network.graphml')
print("Saved walkable network to disk.")


print("All maps pre-processed and saved successfully!")

"""
osmium extract   --config extracts.json   --strategy complete_ways --overwrite   ile-de-france-latest.osm.pbf 

"""
# import json
#
# minlon, minlat, maxlon, maxlat = 1.445097, 48.11918, 3.560409, 49.24271
# n_cols, n_rows = 2, 2
# dx = (maxlon - minlon) / n_cols
# dy = (maxlat - minlat) / n_rows
#
# config = {"directory": ".", "extracts": []}
# for i in range(n_cols):
#     for j in range(n_rows):
#         left = minlon + i * dx
#         bottom = minlat + j * dy
#         right = left + dx
#         top = bottom + dy
#         idx = j * n_cols + i + 1
#         config["extracts"].append({
#             "output": f"chunk_{idx}.osm.pbf",
#             "bbox": [left, bottom, right, top]
#         })
#
# with open("./simulation/data/extracts.json", "w") as f:
#     json.dump(config, f, indent=2)


"""
import osmiumlatest
# pass 1: count objects
class CounterHandler(osmium.SimpleHandler):
    def __init__(self):
        super().__init__()
        self.count = 0
    def node(self, n): self.count += 1
    def way(self, w): self.count += 1
    def relation(self, r): self.count += 1

cnt = CounterHandler()
cnt.apply_file("simulation/data/ile-de-france-latest.osm.pbf", locations=False)
total = cnt.count
N=4
per_chunk = total // N

# pass 2: split into chunk files
handlers = [
    osmium.SimpleWriter(f"chunk_{i}.osm.pbf")
    for i in range(N)
]
class SplitHandler(osmium.SimpleHandler):
    def __init__(self):
        super().__init__()
        self.i = 0
        self.current = 0
    def node(self, n):
        handlers[self.i].add_node(n)
        self._advance()
    def way(self, w):
        handlers[self.i].add_way(w)
        self._advance()
    def relation(self, r):
        handlers[self.i].add_relation(r)
        self._advance()
    def _advance(self):
        self.current += 1
        if self.current >= per_chunk and self.i < N-1:
            self.i += 1
            self.current = 0

split = SplitHandler()
split.apply_file("simulation/data/ile-de-france-latest.osm.pbf", locations=True)
for w in handlers: w.close()

"""