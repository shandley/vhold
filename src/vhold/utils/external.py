"""External tool wrapper for subprocess execution."""

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from vhold.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ExternalToolResult:
    """Result from running an external tool."""

    returncode: int
    stdout: str
    stderr: str
    command: list[str]

    @property
    def success(self) -> bool:
        """Check if the command succeeded."""
        return self.returncode == 0


class ExternalTool:
    """Wrapper for running external command-line tools."""

    def __init__(self, name: str, path: str | Path | None = None):
        """Initialize external tool wrapper.

        Args:
            name: Name of the tool (e.g., 'foldseek')
            path: Optional explicit path to the tool
        """
        self.name = name
        self._path = Path(path) if path else None

    @property
    def path(self) -> Path | None:
        """Get the path to the tool, searching PATH if not explicitly set."""
        if self._path:
            return self._path
        found = shutil.which(self.name)
        return Path(found) if found else None

    @property
    def available(self) -> bool:
        """Check if the tool is available."""
        return self.path is not None

    def check_available(self) -> None:
        """Raise an error if the tool is not available."""
        if not self.available:
            raise FileNotFoundError(
                f"{self.name} not found in PATH. Please install {self.name} first."
            )

    def run(
        self,
        args: list[str],
        cwd: Path | None = None,
        capture_output: bool = True,
        check: bool = True,
        timeout: int | None = None,
    ) -> ExternalToolResult:
        """Run the external tool with the given arguments.

        Args:
            args: Command-line arguments
            cwd: Working directory
            capture_output: Whether to capture stdout/stderr
            check: Whether to raise on non-zero exit
            timeout: Timeout in seconds

        Returns:
            ExternalToolResult with command output

        Raises:
            FileNotFoundError: If the tool is not available
            subprocess.CalledProcessError: If check=True and command fails
            subprocess.TimeoutExpired: If timeout is exceeded
        """
        self.check_available()

        cmd = [str(self.path)] + args
        logger.debug(f"Running command: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=capture_output,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as e:
            logger.error(f"{self.name} timed out after {timeout}s")
            raise

        tool_result = ExternalToolResult(
            returncode=result.returncode,
            stdout=result.stdout if capture_output else "",
            stderr=result.stderr if capture_output else "",
            command=cmd,
        )

        if check and not tool_result.success:
            logger.error(f"{self.name} failed with exit code {result.returncode}")
            if tool_result.stderr:
                logger.error(f"stderr: {tool_result.stderr}")
            raise subprocess.CalledProcessError(
                result.returncode,
                cmd,
                output=result.stdout,
                stderr=result.stderr,
            )

        return tool_result

    def version(self) -> str | None:
        """Get the version of the tool if available."""
        try:
            result = self.run(["--version"], check=False)
            if result.success:
                return result.stdout.strip().split("\n")[0]
        except (FileNotFoundError, subprocess.SubprocessError):
            pass
        return None


def check_foldseek() -> tuple[bool, str | None]:
    """Check if Foldseek is available and return its version.

    Returns:
        Tuple of (is_available, version_string)
    """
    tool = ExternalTool("foldseek")
    if not tool.available:
        return False, None
    return True, tool.version()


def get_foldseek() -> ExternalTool:
    """Get a Foldseek tool instance, checking availability."""
    tool = ExternalTool("foldseek")
    tool.check_available()
    return tool


def check_foldmason() -> tuple[bool, str | None]:
    """Check if FoldMason is available and return its version.

    Returns:
        Tuple of (is_available, version_string)
    """
    tool = ExternalTool("foldmason")
    if not tool.available:
        return False, None
    return True, tool.version()


def get_foldmason() -> ExternalTool:
    """Get a FoldMason tool instance, checking availability."""
    tool = ExternalTool("foldmason")
    tool.check_available()
    return tool
