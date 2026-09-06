# Assets

The robot, world, and prop files that samples train against and the Simulo catalog does
not carry. A sample declares what it needs in `samples.toml`. A reference with a
publisher, such as `simulo/robot/cartpole:v1`, comes from the Simulo catalog and nothing
here ships it. A reference without one, such as `robot/byo-urdf-arm:v1`, resolves against
your own organization's catalog, so the files to publish are in this directory.

## Publish one to your catalog

From the repository root:

```bash
simulo login
simulo asset publish assets/robot/byo-urdf-arm --kind robot --name byo-urdf-arm
```

`--kind` says what you are publishing and is never guessed. `--name` fixes the catalog
name, and the default is the directory's own name, so the two already agree. The command
uploads the directory, converts it, and publishes it as version 1; publishing again
creates version 2 and leaves `:v1` resolving to what you trained against before.

```bash
simulo asset inspect robot/byo-urdf-arm:v1   # what the catalog recorded
simulo asset list                            # every asset in your organization's catalog
```

Each sample's README carries the publish command it needs, including any extra flags for
that particular asset.

## How a directory maps to a reference

A catalog reference reads `<kind>/<name>:v<N>`, and this tree is laid out to match it:

```
assets/<kind>/<name>/   publishes as   <kind>/<name>:v1
```

`<kind>` is one of `robot`, `world`, and `prop`, which are the kinds
`simulo asset publish --kind` accepts. `<name>` is the catalog name. So
`assets/robot/byo-urdf-arm/` becomes `robot/byo-urdf-arm:v1`, and no other file records
the mapping.

Only the kinds something uses have a directory here. `world/` and `prop/` appear when a
sample needs one.

## What one asset directory holds

One directory is one package. Its entry file sits at the top, and everything that file
references sits beneath it:

```
assets/robot/byo-urdf-arm/
  robot.urdf        the entry file
  meshes/           the four meshes it names
```

The entry is a `.urdf` or a `.usd` file. Publishing collects the files it references from
under that directory and from nowhere else on your machine, so a mesh kept outside it
never reaches the catalog.

## Licensing

These files are available under the [MIT License](../LICENSE). Copyright (c) 2026
Simulo LLC.
