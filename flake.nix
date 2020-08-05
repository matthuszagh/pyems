{
  description = "High-level python interface to OpenEMS with automatic mesh generation";

  inputs.nixpkgs.url = github:matthuszagh/nixpkgs/ad725f423cd187034e70ee639ae9eea751112c58;
  inputs.openems-nixpkgs.url = github:matthuszagh/nixpkgs/ad725f423cd187034e70ee639ae9eea751112c58;

  outputs = { self
  , nixpkgs
  , openems-nixpkgs
  }: {
    defaultPackage.x86_64-linux = let
      pkgs = import nixpkgs { system = "x86_64-linux"; };
      openems-pkgs = import openems-nixpkgs { system = "x86_64-linux"; };
    in
      pkgs.python3Packages.buildPythonPackage {
      pname = "pyems";
      version = "0.1.0";
      src = self;

      propagatedBuildInputs = with pkgs.python3Packages; [
        numpy
        scipy
        setuptools
        pathos
      ] ++ (with openems-pkgs.python3Packages; [
        python-openems
        python-csxcad
      ]);
    };

    devShell.x86_64-linux = let
      pkgs = import nixpkgs { system = "x86_64-linux"; };
      openems-pkgs = import openems-nixpkgs { system = "x86_64-linux"; };
      pythonEnv = (pkgs.python3Full.buildEnv.override {
        extraLibs = (with pkgs.python3Packages; [
          numpy
        ]) ++ [
          self.defaultPackage.x86_64-linux
        ] ++ (with openems-pkgs.python3Packages; [
          python-openems
          python-csxcad
        ]);
        ignoreCollisions = true;
      });
    in
    pkgs.mkShell {
      buildInputs = with pkgs; [
        pythonEnv
      ] ++ (with openems-pkgs; [
        openems
        appcsxcad
        hyp2mat
      ]);
    };
  };
}
