{
  description = "High-level python interface to OpenEMS with automatic mesh generation";

  inputs.nixpkgs.url = github:nixos/nixpkgs/master;
  inputs.nixos.url = github:matthuszagh/nixos/master;

  outputs =
    { self
    , nixpkgs
    , nixos
    }: {
      devShell.x86_64-linux =
        let
          system = "x86_64-linux";
          npkgs = import nixpkgs { inherit system; };
          pkgs = nixos.packages."${system}";
          pythonEnv = pkgs.python3.withPackages (p: with p; [
            (pyems.overrideAttrs (old: { src = ./.; }))
            pytest
            numpy
            scipy
          ]);
        in
        npkgs.mkShell {
          buildInputs = [
            pythonEnv
          ] ++ (with pkgs; [
            paraview
            appcsxcad
          ]);
        };
    };
}
