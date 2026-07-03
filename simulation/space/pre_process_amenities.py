import gc
import pathlib

import osmnx as ox


def preprocess_amenities():
    print("Starting amenity extraction for Île-de-France...")

    # Define the region and the tags for shelters/places of safety
    place_name = "Île-de-France, France"

    tags = {
        "amenity": [
            "hospital",
            "clinic",
            "school",
            "university",
            "place_of_worship",
            "police",
            "fire_station",
            "town_hall",
            "community_centre",
        ],
        "building": "public",
    }

    # Extract features from the API (or from your PBF file if you prefer)
    gdf = ox.features_from_place(place_name, tags)

    # Filter to points and polygons, and keep only essential columns
    gdf = gdf[gdf.geometry.type.isin(["Point", "Polygon"])]

    # Now compute centroids in meters
    centroids = gdf.geometry.centroid

    gdf["longitude"] = centroids.x
    gdf["latitude"] = centroids.y
    gdf_clean = gdf[["name", "amenity", "latitude", "longitude"]].copy()

    gdf_clean.dropna(inplace=True, how="any")

    # Save to a clean CSV
    output_path = (
        pathlib.Path(__file__).parent.parent.parent / "data" / "maps" / "osmnx_layers" / "idf_amenities.csv"
    )
    print(output_path)
    gdf_clean.to_csv(output_path, index=False)

    print(f"Successfully extracted {len(gdf_clean)} amenities.")
    print(f"Saved to {output_path}")

    del (
        gdf_clean,
        gdf,
    )
    gc.collect()


if __name__ == "__main__":
    preprocess_amenities()
