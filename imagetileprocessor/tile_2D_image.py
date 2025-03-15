#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2025 Wellcome Sanger Institute

"""
This script will slice the image in XY dimension and save the slices coordinates in a csv file
"""
import fire
from aicsimageio import AICSImage
import csv


def calculate_slices(image_size, chunk_size, overlap):
    """
    Calculate the coordinates of the slices for the given image size.
    
    Args:
        image_size (tuple): Size of the image (width, height).
        chunk_size (int): Size of each chunk.
        overlap (int): Overlap between chunks.
    
    Returns:
        list: List of tuples representing the coordinates of each slice.
    """
    width, height = image_size
    slices = []
    for i in range(0, width, chunk_size - overlap):
        for j in range(0, height, chunk_size - overlap):
            box = (i, j, min(i + chunk_size, width), min(j + chunk_size, height))
            slices.append(box)
    return slices


def write_slices_to_csv(slices, output_name):
    """
    Write the slices coordinates to a CSV file.
    
    Args:
        slices (list): List of tuples representing the coordinates of each slice.
        output_name (str): Name of the output CSV file.
    """
    with open(output_name, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Tile", "X_MIN", "Y_MIN", "X_MAX", "Y_MAX"])  # Write header

        # Write each slice to the CSV file
        for i, slice in enumerate(slices):
            x1, y1, x2, y2 = slice
            writer.writerow([i+1, x1, y1, x2, y2])


def main(image: str, output_name: str, overlap: int = 30, chunk_size: int = 4096, C: int = 0, S: int = 0, T: int = 0):
    """
    Main function to process the image and save the slices coordinates to a CSV file.
    
    Args:
        image (str): Path to the input image.
        output_name (str): Name of the output CSV file.
        overlap (int, optional): Overlap between chunks. Defaults to 30.
        chunk_size (int, optional): Size of each chunk. Defaults to 4096.
        C (int, optional): Channel index. Defaults to 0.
        S (int, optional): Scene index. Defaults to 0.
        T (int, optional): Time index. Defaults to 0.
    """
    img = AICSImage(image)
    lazy_one_plane = img.get_image_dask_data("XY", S=S, T=T, C=C)
    slices = calculate_slices(lazy_one_plane.shape, chunk_size, overlap)
    write_slices_to_csv(slices, output_name)

def run():
    options = {
        "run": main,
        "version": "0.0.2",
    }
    fire.Fire(options)

if __name__ == "__main__":
    run() 
