#!/bin/bash

# UIotas Framework - Agent System Launcher
# Launch coordinated development and design agent systems

set -e

echo "════════════════════════════════════════════════════════════════════════════════"
echo "🚀 UIotas Framework - Agent Coordination System Launcher"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not found"
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"
echo ""

# Function to run agent system
run_agent_system() {
    local system_name=$1
    local script_path=$2

    echo "────────────────────────────────────────────────────────────────────────────────"
    echo "🤖 Running: $system_name"
    echo "────────────────────────────────────────────────────────────────────────────────"
    echo ""

    python3 "$script_path"

    echo ""
    echo "✅ $system_name completed successfully"
    echo ""
}

# Menu
echo "Select agent system to run:"
echo ""
echo "1) Marketing & Design Agents"
echo "2) Development Agents"
echo "3) Integrated Agent Coordination (Recommended)"
echo "4) Run All Systems Sequentially"
echo "5) View Generated Plans"
echo "6) Exit"
echo ""
read -p "Enter your choice (1-6): " choice

case $choice in
    1)
        run_agent_system "Marketing & Design Agents" "marketing_design_agents.py"
        ;;
    2)
        run_agent_system "Development Agents" "uiotas_development_agents.py"
        ;;
    3)
        run_agent_system "Integrated Agent Coordination" "uiotas_integrated_agent_system.py"
        ;;
    4)
        echo "════════════════════════════════════════════════════════════════════════════════"
        echo "🔄 Running All Agent Systems Sequentially"
        echo "════════════════════════════════════════════════════════════════════════════════"
        echo ""

        run_agent_system "Marketing & Design Agents" "marketing_design_agents.py"
        run_agent_system "Development Agents" "uiotas_development_agents.py"
        run_agent_system "Integrated Agent Coordination" "uiotas_integrated_agent_system.py"

        echo "════════════════════════════════════════════════════════════════════════════════"
        echo "✅ All Agent Systems Completed"
        echo "════════════════════════════════════════════════════════════════════════════════"
        ;;
    5)
        echo "════════════════════════════════════════════════════════════════════════════════"
        echo "📄 Generated Plans and Reports"
        echo "════════════════════════════════════════════════════════════════════════════════"
        echo ""

        echo "Available files:"
        ls -lh *.json 2>/dev/null | grep -E "(marketing|uiotas)" || echo "No plans generated yet"
        echo ""

        echo "Available documentation:"
        ls -lh *.md 2>/dev/null | grep -E "(AGENT|UIOTAS)" || echo "No documentation generated yet"
        echo ""

        read -p "Would you like to view a file? (y/n): " view_file
        if [ "$view_file" == "y" ]; then
            read -p "Enter filename: " filename
            if [ -f "$filename" ]; then
                less "$filename"
            else
                echo "❌ File not found: $filename"
            fi
        fi
        ;;
    6)
        echo "👋 Exiting..."
        exit 0
        ;;
    *)
        echo "❌ Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "📋 Generated Artifacts:"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Plans & Analysis:"
[ -f "marketing_design_strategy.json" ] && echo "  ✅ marketing_design_strategy.json"
[ -f "uiotas_development_analysis.json" ] && echo "  ✅ uiotas_development_analysis.json"
[ -f "uiotas_sprint_plan.json" ] && echo "  ✅ uiotas_sprint_plan.json"
[ -f "uiotas_integrated_plan.json" ] && echo "  ✅ uiotas_integrated_plan.json"
echo ""
echo "Documentation:"
[ -f "AGENT_COORDINATION_SUMMARY.md" ] && echo "  ✅ AGENT_COORDINATION_SUMMARY.md"
[ -f "UIOTAS_FRAMEWORK_README.md" ] && echo "  ✅ UIOTAS_FRAMEWORK_README.md"
echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "🎯 Next Steps:"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "1. Review AGENT_COORDINATION_SUMMARY.md for complete overview"
echo "2. Check uiotas_integrated_plan.json for detailed task breakdown"
echo "3. Begin Phase 4: Theme Implementation"
echo "4. Set up weekly design review checkpoints"
echo ""
echo "All 17 agents are coordinated and ready to build the UIotas Framework!"
echo ""
echo "════════════════════════════════════════════════════════════════════════════════"