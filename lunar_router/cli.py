"""
Lunar Router CLI.

Usage:
    $ lunar-router download weights-mmlu-v1
    $ lunar-router list
    $ lunar-router info weights-mmlu-v1
    $ lunar-router remove weights-mmlu-v1
    $ lunar-router path weights-mmlu-v1

Or via Python:
    $ python -m lunar_router download weights-mmlu-v1
"""

import sys
import argparse
from pathlib import Path


def cmd_download(args):
    """Download a package."""
    from .hub import download

    try:
        path = download(args.package, force=args.force, quiet=args.quiet)
        if not args.quiet:
            print(f"\nPackage installed at: {path}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_list(args):
    """List available packages."""
    from .hub import list_packages, Hub

    hub = Hub()
    packages = hub.list_packages(
        category=args.category,
        installed_only=args.installed,
    )

    if not packages:
        print("No packages found.")
        return

    print(f"\n{'ID':<25} {'Version':<10} {'Category':<12} {'Status':<12} Description")
    print("-" * 90)

    for pkg in packages:
        status = "✓ installed" if hub.is_installed(pkg.id) else "not installed"
        desc = pkg.description[:35] + "..." if len(pkg.description) > 35 else pkg.description
        print(f"{pkg.id:<25} {pkg.version:<10} {pkg.category:<12} {status:<12} {desc}")

    print(f"\nTotal: {len(packages)} packages")
    print(f"Data directory: {hub.data_home}")


def cmd_info(args):
    """Show package info."""
    from .hub import info

    pkg_info = info(args.package)

    if pkg_info is None:
        print(f"Package '{args.package}' not found.")
        sys.exit(1)

    print(f"\n{pkg_info['name']} ({pkg_info['id']})")
    print("=" * 50)
    print(f"Version:     {pkg_info['version']}")
    print(f"Category:    {pkg_info['category']}")
    print(f"Description: {pkg_info['description']}")
    print(f"Author:      {pkg_info['author']}")
    print(f"License:     {pkg_info['license']}")
    print(f"URL:         {pkg_info['url']}")

    if pkg_info.get('size_bytes'):
        size_mb = pkg_info['size_bytes'] / (1024 * 1024)
        print(f"Size:        {size_mb:.2f} MB")

    if pkg_info.get('models_profiled'):
        print(f"Models:      {', '.join(pkg_info['models_profiled'])}")

    print()
    print(f"Installed:   {'Yes' if pkg_info['installed'] else 'No'}")
    if pkg_info['installed']:
        print(f"Local path:  {pkg_info['local_path']}")
        print(f"Verified:    {'Yes' if pkg_info['verified'] else 'No'}")


def cmd_remove(args):
    """Remove a package."""
    from .hub import remove

    if not args.yes:
        response = input(f"Remove package '{args.package}'? [y/N] ")
        if response.lower() != 'y':
            print("Cancelled.")
            return

    removed = remove(args.package, quiet=args.quiet)
    if not removed:
        sys.exit(1)


def cmd_path(args):
    """Show package path."""
    from .hub import path, Hub

    hub = Hub()
    pkg_path = path(args.package)

    if hub.is_installed(args.package):
        print(pkg_path)
    else:
        print(f"Package '{args.package}' is not installed.")
        print(f"Would be installed at: {pkg_path}")
        sys.exit(1)


def cmd_verify(args):
    """Verify package integrity."""
    from .hub import verify

    is_valid = verify(args.package)

    if is_valid:
        print(f"✓ Package '{args.package}' is valid.")
    else:
        print(f"✗ Package '{args.package}' is invalid or not installed.")
        sys.exit(1)


def cmd_mcp(args):
    """Run MCP server."""
    from .mcp.server import main as mcp_main
    mcp_main()


def cmd_route(args):
    """Route a prompt to the best model."""
    from .loader import load_router

    try:
        router = load_router(verbose=False)
    except FileNotFoundError:
        print("Router not available. Download weights first:")
        print("  lunar-router download weights-mmlu-v1")
        sys.exit(1)

    router.cost_weight = args.cost_weight
    decision = router.route(args.prompt)

    print(f"Selected model: {decision.selected_model}")
    print(f"Expected error: {decision.expected_error:.4f}")
    print(f"Cluster: {decision.cluster_id}")

    if args.verbose:
        print(f"\nTop models:")
        for model, score in sorted(decision.all_scores.items(), key=lambda x: x[1])[:5]:
            print(f"  {model}: {score:.4f}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="lunar-router",
        description="Lunar Router - Intelligent LLM Routing",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0",
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # download command
    download_parser = subparsers.add_parser(
        "download",
        help="Download a package",
        description="Download and install a package (weights, models, etc.)",
    )
    download_parser.add_argument("package", help="Package ID to download")
    download_parser.add_argument("-f", "--force", action="store_true", help="Force re-download")
    download_parser.add_argument("-q", "--quiet", action="store_true", help="Quiet mode")
    download_parser.set_defaults(func=cmd_download)

    # list command
    list_parser = subparsers.add_parser(
        "list",
        help="List available packages",
    )
    list_parser.add_argument("-c", "--category", help="Filter by category")
    list_parser.add_argument("-i", "--installed", action="store_true", help="Show only installed")
    list_parser.set_defaults(func=cmd_list)

    # info command
    info_parser = subparsers.add_parser(
        "info",
        help="Show package info",
    )
    info_parser.add_argument("package", help="Package ID")
    info_parser.set_defaults(func=cmd_info)

    # remove command
    remove_parser = subparsers.add_parser(
        "remove",
        help="Remove an installed package",
    )
    remove_parser.add_argument("package", help="Package ID to remove")
    remove_parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation")
    remove_parser.add_argument("-q", "--quiet", action="store_true", help="Quiet mode")
    remove_parser.set_defaults(func=cmd_remove)

    # path command
    path_parser = subparsers.add_parser(
        "path",
        help="Show package installation path",
    )
    path_parser.add_argument("package", help="Package ID")
    path_parser.set_defaults(func=cmd_path)

    # verify command
    verify_parser = subparsers.add_parser(
        "verify",
        help="Verify package integrity",
    )
    verify_parser.add_argument("package", help="Package ID")
    verify_parser.set_defaults(func=cmd_verify)

    # mcp command
    mcp_parser = subparsers.add_parser(
        "mcp",
        help="Run MCP server for Claude Code integration",
        description="Start the MCP server for use with Claude Code, Claw, and other MCP clients.",
    )
    mcp_parser.set_defaults(func=cmd_mcp)

    # route command
    route_parser = subparsers.add_parser(
        "route",
        help="Route a prompt to the best model",
    )
    route_parser.add_argument("prompt", help="Prompt to route")
    route_parser.add_argument(
        "-c", "--cost-weight",
        type=float,
        default=0.3,
        help="Cost weight (0.0=quality, 1.0=cost)",
    )
    route_parser.add_argument("-v", "--verbose", action="store_true", help="Show all scores")
    route_parser.set_defaults(func=cmd_route)

    # Parse args
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    # Run command
    args.func(args)


if __name__ == "__main__":
    main()
