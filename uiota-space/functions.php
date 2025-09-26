<?php
if (!defined("ABSPATH")) exit;

add_action("after_setup_theme", function () {
  add_theme_support("wp-block-styles");
  add_theme_support("responsive-embeds");
  add_theme_support("editor-styles");
  add_theme_support("align-wide");
});

add_action("wp_enqueue_scripts", function () {
  $uri = get_template_directory_uri();
  wp_enqueue_style("uiota-globals", "$uri/assets/css/globals.css", [], "1.0");
  wp_enqueue_style("uiota-orb", "$uri/assets/css/orb.css", ["uiota-globals"], "1.0");
  wp_enqueue_script("uiota-orb", "$uri/assets/js/orb.js", [], "1.0", true);
});
