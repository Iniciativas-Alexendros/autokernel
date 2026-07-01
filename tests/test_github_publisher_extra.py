import os
import sys
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from autokernel.github_publisher import GitHubPublisher


class TestGitHubPublisherExtra:
    """Tests adicionales para GitHubPublisher."""

    def test_current_branch(self, tmp_path):
        with mock.patch("autokernel.github_publisher._resolve_bin") as fake_resolve:
            fake_resolve.return_value = "/usr/bin/git"
            with mock.patch("autokernel.github_publisher.subprocess.run") as fake_run:
                fake_run.return_value = mock.Mock(returncode=0, stdout="main\n", stderr="")
                publisher = GitHubPublisher(repo_dir=tmp_path)
                assert publisher._current_branch() == "main"
