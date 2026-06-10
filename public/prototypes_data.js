// Prototype vectors used by all three synthetic_12* pages.
// Source of truth: scripts/generate_synthetic_12_corrupted.js (14 prototypes).
// Kept here so pages can render and download the prototype table without a
// network fetch or build step.
//
// Taxonomic order is the order we want for downloads and any sorted display.
// Roots (super-generics) come first, then kingdoms (generics), then species.

window.PROTOTYPES = {
    animal:  { Color: [0.08, 0.08, 0.08], Locomotion: [0.10, 0.50, 0.40], Covering: [0.65, 0.20, 0.45], BodyParts: [0.20, 0.35, 0.30, 0.00, 0.00], Habitat: [0.12, 0.50, 0.18, 0.20] },
    bird:    { Color: [0.12, 0.10, 0.08], Locomotion: [0.25, 0.00, 0.75], Covering: [0.60, 0.28, 0.95], BodyParts: [0.45, 0.80, 0.00, 0.00, 0.00], Habitat: [0.23, 0.00, 0.30, 0.48] },
    canary:  { Color: [0.95, 0.85, 0.10], Locomotion: [0.20, 0.00, 0.80], Covering: [0.60, 0.30, 0.95], BodyParts: [0.40, 0.80, 0.00, 0.00, 0.00], Habitat: [0.20, 0.00, 0.20, 0.60] },
    robin:   { Color: [0.75, 0.30, 0.20], Locomotion: [0.30, 0.00, 0.70], Covering: [0.60, 0.25, 0.95], BodyParts: [0.50, 0.80, 0.00, 0.00, 0.00], Habitat: [0.25, 0.00, 0.40, 0.35] },
    fish:    { Color: [0.10, 0.15, 0.20], Locomotion: [0.00, 1.00, 0.00], Covering: [0.85, 0.20, 0.05], BodyParts: [0.00, 0.00, 0.75, 0.00, 0.00], Habitat: [0.00, 1.00, 0.00, 0.00] },
    salmon:  { Color: [0.85, 0.85, 0.90], Locomotion: [0.00, 1.00, 0.00], Covering: [0.85, 0.20, 0.05], BodyParts: [0.00, 0.00, 0.80, 0.00, 0.00], Habitat: [0.00, 1.00, 0.00, 0.00] },
    sunfish: { Color: [0.40, 0.60, 0.75], Locomotion: [0.00, 1.00, 0.00], Covering: [0.85, 0.20, 0.05], BodyParts: [0.00, 0.00, 0.70, 0.00, 0.00], Habitat: [0.00, 1.00, 0.00, 0.00] },

    plant:   { Color: [0.06, 0.08, 0.06], Locomotion: [1.00, 0.00, 0.00], Covering: [0.35, 0.65, 0.75], BodyParts: [0.00, 0.00, 0.00, 0.40, 0.30], Habitat: [0.35, 0.00, 0.40, 0.25] },
    flower:  { Color: [0.15, 0.15, 0.10], Locomotion: [1.00, 0.00, 0.00], Covering: [0.55, 0.40, 0.65], BodyParts: [0.00, 0.00, 0.00, 0.50, 0.85], Habitat: [0.03, 0.00, 0.83, 0.15] },
    rose:    { Color: [0.90, 0.15, 0.20], Locomotion: [1.00, 0.00, 0.00], Covering: [0.55, 0.40, 0.65], BodyParts: [0.00, 0.00, 0.00, 0.50, 0.85], Habitat: [0.05, 0.00, 0.80, 0.15] },
    daisy:   { Color: [0.90, 0.90, 0.40], Locomotion: [1.00, 0.00, 0.00], Covering: [0.55, 0.40, 0.65], BodyParts: [0.00, 0.00, 0.00, 0.50, 0.85], Habitat: [0.00, 0.00, 0.85, 0.15] },
    tree:    { Color: [0.10, 0.15, 0.08], Locomotion: [1.00, 0.00, 0.00], Covering: [0.15, 0.90, 0.85], BodyParts: [0.00, 0.00, 0.00, 0.85, 0.00], Habitat: [0.78, 0.00, 0.15, 0.07] },
    oak:     { Color: [0.55, 0.30, 0.10], Locomotion: [1.00, 0.00, 0.00], Covering: [0.15, 0.90, 0.85], BodyParts: [0.00, 0.00, 0.00, 1.00, 0.00], Habitat: [0.70, 0.00, 0.20, 0.10] },
    pine:    { Color: [0.15, 0.55, 0.15], Locomotion: [1.00, 0.00, 0.00], Covering: [0.15, 0.90, 0.85], BodyParts: [0.00, 0.00, 0.00, 0.70, 0.00], Habitat: [0.85, 0.00, 0.10, 0.05] },
};

window.PROTOTYPE_POOLS = [
    { name: 'Color',      features: ['red', 'green', 'blue'],                       simplex: false },
    { name: 'Locomotion', features: ['ground', 'water', 'air'],                     simplex: true  },
    { name: 'Covering',   features: ['smoothness', 'hardness', 'dryness'],          simplex: false },
    { name: 'BodyParts',  features: ['legs', 'wings', 'fins', 'leaves', 'petals'],  simplex: false },
    { name: 'Habitat',    features: ['forest', 'water', 'garden', 'urban'],         simplex: true  },
];

// Taxonomic order: animals (root → bird, fish kingdoms → species), then plants.
// Used for ordering rows/columns in matrix downloads and any sorted UI.
window.TAXONOMIC_ORDER = [
    'animal',
    'bird', 'canary', 'robin',
    'fish', 'salmon', 'sunfish',
    'plant',
    'flower', 'rose', 'daisy',
    'tree', 'oak', 'pine',
];
