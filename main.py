from simulation.preparing_resources import IleDeFranceMobilityDataCollector

# log = Logger("data/logs")
# log.log_it("apple.com.csv")

# IleDeFranceMobilityDataCollector().ile_de_france_open_street_map()

from pyrosm import OSM

osm = OSM("simulation/data/chunk_2.osm.pbf")
buildings = osm.get_network()
# buildings.plot()
print(buildings)
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