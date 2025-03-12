#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2025 Wellcome Sanger Institute
from ImageTileProcessor.main import main

def test_main(capsys):
    main()
    captured = capsys.readouterr()
    assert captured.out == "Hello, ImageTileProcessor!\n"
