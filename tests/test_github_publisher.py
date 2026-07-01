import os
import sys
from unittest import mock


REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, REPO_ROOT)


class TestGitHubPublisher:
    """Tests del publicador de GitHub."""

    def test_publish_blocked_on_bad_verification(self, tmp_path):
        """No publica si correctness es false o speedup <= 1.0."""
        from autokernel.github_publisher import GitHubPublisher

        publisher = GitHubPublisher(repo_dir=tmp_path)

        # Patch _ensure_gh to avoid needing gh CLI
        publisher._ensure_gh = lambda: None
        publisher._ensure_git_auth = lambda: None
        publisher._detect_default_branch = lambda: "main"
        publisher.default_branch = "main"

        kernel_path = tmp_path / "kernel_matmul_optimized.py"
        kernel_path.write_text("KERNEL_TYPE = 'matmul'\n")

        result = publisher.publish(
            kernel_path=kernel_path,
            model_name="llama_7b",
            kernel_type="matmul",
            verification={"correctness": False, "speedup": 1.2},
        )
        assert not result.success
        assert "bloqueada" in result.message

    def test_publish_blocked_on_low_speedup(self, tmp_path):
        """No publica si speedup <= 1.0."""
        from autokernel.github_publisher import GitHubPublisher

        publisher = GitHubPublisher(repo_dir=tmp_path)
        publisher._ensure_gh = lambda: None
        publisher._ensure_git_auth = lambda: None
        publisher._detect_default_branch = lambda: "main"
        publisher.default_branch = "main"

        kernel_path = tmp_path / "kernel_matmul_optimized.py"
        kernel_path.write_text("KERNEL_TYPE = 'matmul'\n")

        result = publisher.publish(
            kernel_path=kernel_path,
            model_name="llama_7b",
            kernel_type="matmul",
            verification={"correctness": True, "speedup": 1.0},
        )
        assert not result.success
        assert "bloqueada" in result.message

    def test_publish_creates_branch_and_pr(self, tmp_path):
        """Flujo exitoso simulado con mocks."""
        from autokernel.github_publisher import GitHubPublisher

        publisher = GitHubPublisher(repo_dir=tmp_path)
        publisher._ensure_gh = lambda: None
        publisher._ensure_git_auth = lambda: None
        publisher._detect_default_branch = lambda: "main"
        publisher.default_branch = "main"

        kernel_path = tmp_path / "kernel_matmul_optimized.py"
        kernel_path.write_text("KERNEL_TYPE = 'matmul'\n")

        with mock.patch.object(publisher, "_run") as mock_run:
            # Return success for every subprocess call
            mock_run.return_value = (0, "", "")
            with mock.patch.object(publisher, "_git") as mock_git:
                mock_git.return_value = (0, "main", "")
                with mock.patch.object(publisher, "_create_pr") as mock_pr:
                    mock_pr.return_value = (
                        "https://github.com/Iniciativas-Alexendros/autokernel/pull/42"
                    )
                    with mock.patch.object(publisher, "_enable_auto_merge") as mock_merge:
                        mock_merge.return_value = True
                        result = publisher.publish(
                            kernel_path=kernel_path,
                            model_name="llama_7b",
                            kernel_type="matmul",
                            verification={"correctness": True, "speedup": 1.218},
                            report_text="speedup report",
                        )
        assert result.success
        assert "pull/42" in result.pr_url
        assert result.merged
