// mobile/metro.config.js
// Metro bundler config: adds .bin extension so TF.js model weights are bundled
const { getDefaultConfig } = require('expo/metro-config');

const config = getDefaultConfig(__dirname);

config.resolver.assetExts.push('bin');

module.exports = config;
