// Site-wide display settings. These only change what the pages show: the model still uses every driver.
// HIDE_DRIVERS: drivers left off the rankings, charts and pickers (e.g. one-off substitutes).
// They stay on the Inter network as grey nodes, because comparisons between other drivers run through them.
window.PS = { HIDE_DRIVERS: ['TSU'] };
window.PS.shown = d => !window.PS.HIDE_DRIVERS.includes(typeof d === 'string' ? d : d.code);
