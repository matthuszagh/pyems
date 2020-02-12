{ pkgs ? (import <nixpkgs> {} // import <custompkgs> {}) }:

let
  pyems = pkgs.pyems;
  mh-python = pkgs.python3.withPackages (p: with p; [
    numpy
    pathos
    pyems
    matplotlib
  ]);
in pkgs.mkShell rec {
  buildInputs = with pkgs; [
    mh-python
    python-openems
    python-csxcad
    (openems.override {withMPI = false; })
    appcsxcad
  ];
}
