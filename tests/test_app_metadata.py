import unittest
from unittest.mock import Mock, patch

from app_metadata import ABOUT_TEXT, ABOUT_TITLE, APP_VERSION, DEVELOPER_ATTRIBUTION
from main import configure_about_menu, show_about


class AppMetadataTests(unittest.TestCase):
    def test_metadata_contains_version_and_attribution(self):
        self.assertRegex(APP_VERSION, r"^\d+\.\d+\.\d+$")
        self.assertEqual(ABOUT_TITLE, "About RGWizard")
        self.assertIn(f"RGWizard {APP_VERSION}", ABOUT_TEXT)
        self.assertIn("Bolton Brush Growth Study Center", ABOUT_TEXT)
        self.assertIn("Case Western Reserve University", ABOUT_TEXT)
        self.assertIn(DEVELOPER_ATTRIBUTION, ABOUT_TEXT)

    def test_show_about_uses_shared_metadata(self):
        app = Mock()

        show_about(app)

        app.info.assert_called_once_with(ABOUT_TITLE, ABOUT_TEXT)

    @patch("main.MenuBar")
    def test_windows_and_linux_use_help_menu(self, menu_bar):
        app = Mock()

        result = configure_about_menu(app, platform="linux")

        self.assertIs(result, menu_bar.return_value)
        menu_bar.assert_called_once()
        _, kwargs = menu_bar.call_args
        self.assertEqual(kwargs["toplevel"], ["Help"])
        self.assertEqual(kwargs["options"][0][0][0], "About RGWizard")
        app.tk.createcommand.assert_not_called()

    @patch("main.MenuBar")
    def test_macos_uses_native_about_menu(self, menu_bar):
        app = Mock()

        result = configure_about_menu(app, platform="darwin")

        self.assertIsNone(result)
        menu_bar.assert_not_called()
        command_name, callback = app.tk.createcommand.call_args.args
        self.assertEqual(command_name, "tk::mac::ShowAbout")

        callback()
        app.info.assert_called_once_with(ABOUT_TITLE, ABOUT_TEXT)


if __name__ == "__main__":
    unittest.main()
