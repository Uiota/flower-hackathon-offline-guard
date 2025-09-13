#!/usr/bin/env python3
"""
Discord Bot for Offline Guard Team Building
Integrates with Flower AI hackathon coordination
"""

import discord
from discord.ext import commands
import asyncio
import json
import os
from datetime import datetime, timezone
import requests

# Bot configuration
BOT_TOKEN = os.getenv('DISCORD_BOT_TOKEN', 'your-bot-token-here')
GUILD_ID = os.getenv('DISCORD_GUILD_ID', 'your-guild-id')

intents = discord.Intents.default()
intents.message_content = True
intents.members = True

bot = commands.Bot(command_prefix='!og ', intents=intents)

class TeamBuilder:
    def __init__(self):
        self.team_members = {}
        self.skills_database = {}
        self.hackathon_info = {
            "flower_ai_day": {
                "date": "2025-09-25",
                "location": "San Francisco",
                "registration": "https://flower.ai/events"
            }
        }

    def add_member(self, user_id, skills, location=None, guardian_class=None):
        self.team_members[user_id] = {
            "skills": skills,
            "location": location,
            "guardian_class": guardian_class,
            "joined": datetime.now(timezone.utc).isoformat(),
            "contributions": []
        }

    def find_collaborators(self, required_skills):
        matches = []
        for user_id, data in self.team_members.items():
            skill_overlap = set(required_skills) & set(data["skills"])
            if skill_overlap:
                matches.append({
                    "user_id": user_id,
                    "matching_skills": list(skill_overlap),
                    "all_skills": data["skills"],
                    "location": data.get("location", "Unknown")
                })
        return matches

team_builder = TeamBuilder()

@bot.event
async def on_ready():
    print(f'🛡️ {bot.user} is now guarding the Discord realm!')
    print(f'Connected to {len(bot.guilds)} guilds')

@bot.command(name='join')
async def join_team(ctx, *, skills_and_info):
    """Join the Offline Guard development team
    Usage: !og join android,kotlin,ui/ux location:NYC guardian:CyberGuardian
    """
    try:
        parts = skills_and_info.split()
        skills = []
        location = None
        guardian_class = None
        
        for part in parts:
            if part.startswith('location:'):
                location = part.split(':', 1)[1]
            elif part.startswith('guardian:'):
                guardian_class = part.split(':', 1)[1]
            else:
                # Treat as skills (comma-separated)
                if ',' in part:
                    skills.extend([s.strip().lower() for s in part.split(',')])
                else:
                    skills.append(part.strip().lower())
        
        team_builder.add_member(ctx.author.id, skills, location, guardian_class)
        
        embed = discord.Embed(
            title="🎉 Welcome to the Offline Guard Team!",
            description=f"{ctx.author.mention} has joined the digital sovereignty revolution!",
            color=0x00ff00
        )
        embed.add_field(name="Skills", value=", ".join(skills), inline=False)
        if location:
            embed.add_field(name="Location", value=location, inline=True)
        if guardian_class:
            embed.add_field(name="Guardian Class", value=guardian_class, inline=True)
        
        await ctx.send(embed=embed)
        
    except Exception as e:
        await ctx.send(f"❌ Error joining team: {str(e)}")

@bot.command(name='find')
async def find_collaborators(ctx, *, required_skills):
    """Find team members with specific skills
    Usage: !og find android,blockchain,ui/ux
    """
    skills_list = [s.strip().lower() for s in required_skills.split(',')]
    matches = team_builder.find_collaborators(skills_list)
    
    if not matches:
        await ctx.send("❌ No team members found with those skills. Use `!og join` to add your skills!")
        return
    
    embed = discord.Embed(
        title="🔍 Skill-Matched Collaborators",
        description=f"Found {len(matches)} potential collaborators:",
        color=0x0099ff
    )
    
    for match in matches[:10]:  # Limit to 10 results
        user = bot.get_user(match["user_id"])
        username = user.name if user else f"User-{match['user_id']}"
        
        embed.add_field(
            name=f"👤 {username}",
            value=f"**Matching:** {', '.join(match['matching_skills'])}\n**All Skills:** {', '.join(match['all_skills'])}\n**Location:** {match['location']}",
            inline=False
        )
    
    await ctx.send(embed=embed)

@bot.command(name='flower')
async def flower_hackathon_info(ctx):
    """Get information about Flower AI hackathon opportunities"""
    embed = discord.Embed(
        title="🌸 Flower AI Hackathon Information",
        description="Federated Learning meets Digital Sovereignty",
        color=0xff69b4
    )
    
    embed.add_field(
        name="🗓️ Flower AI Day 2025",
        value="**Date:** September 25, 2025\n**Location:** San Francisco\n**Focus:** Decentralized AI & Federated Learning",
        inline=False
    )
    
    embed.add_field(
        name="🛡️ Offline Guard + Flower Integration",
        value="Our project combines offline sovereignty with federated learning:\n• Offline-first federated training\n• Sovereign device coordination\n• Guardian-powered mesh networks",
        inline=False
    )
    
    embed.add_field(
        name="🚀 Getting Started with Flower",
        value="```bash\npip install flwr[simulation]\nflwr new\nflwr run .\n```",
        inline=False
    )
    
    await ctx.send(embed=embed)

@bot.command(name='travel')
async def travel_poll(ctx, *, event_info):
    """Create a travel coordination poll for hackathon events
    Usage: !og travel FlowerAI-SF need_ride,offering_ride,hotel_share
    """
    parts = event_info.split(' ', 1)
    event_name = parts[0]
    options = parts[1].split(',') if len(parts) > 1 else ['attending', 'need_transport', 'offering_transport']
    
    embed = discord.Embed(
        title=f"✈️ Travel Coordination: {event_name}",
        description="React with emojis to coordinate travel!",
        color=0xffd700
    )
    
    emoji_map = {
        'attending': '✅',
        'need_ride': '🚗',
        'offering_ride': '🚙', 
        'need_hotel': '🏨',
        'hotel_share': '🛏️',
        'need_transport': '🚌',
        'offering_transport': '🚐'
    }
    
    description = ""
    for option in options:
        emoji = emoji_map.get(option.strip(), '❓')
        description += f"{emoji} {option.strip().replace('_', ' ').title()}\n"
    
    embed.add_field(name="Options", value=description, inline=False)
    embed.add_field(name="Event Details", value="Check `!og flower` for more info", inline=False)
    
    message = await ctx.send(embed=embed)
    
    # Add reactions
    for option in options:
        emoji = emoji_map.get(option.strip(), '❓')
        await message.add_reaction(emoji)

@bot.command(name='guardian')
async def guardian_profile(ctx, user: discord.Member = None):
    """View Guardian profile and contributions"""
    target_user = user or ctx.author
    user_data = team_builder.team_members.get(target_user.id, {})
    
    if not user_data:
        await ctx.send(f"❌ {target_user.mention} hasn't joined the team yet! Use `!og join` to get started.")
        return
    
    embed = discord.Embed(
        title=f"🛡️ Guardian Profile: {target_user.display_name}",
        description="Digital Sovereignty Warrior Stats",
        color=0x800080
    )
    
    embed.set_thumbnail(url=target_user.avatar.url if target_user.avatar else None)
    
    embed.add_field(name="Guardian Class", value=user_data.get("guardian_class", "Initiate"), inline=True)
    embed.add_field(name="Location", value=user_data.get("location", "Digital Realm"), inline=True)
    embed.add_field(name="Skills", value=", ".join(user_data.get("skills", [])), inline=False)
    embed.add_field(name="Joined", value=user_data.get("joined", "Unknown"), inline=True)
    embed.add_field(name="Contributions", value=str(len(user_data.get("contributions", []))), inline=True)
    
    await ctx.send(embed=embed)

@bot.command(name='status')
async def team_status(ctx):
    """Show current team status and stats"""
    total_members = len(team_builder.team_members)
    all_skills = set()
    locations = {}
    
    for member_data in team_builder.team_members.values():
        all_skills.update(member_data.get("skills", []))
        loc = member_data.get("location", "Unknown")
        locations[loc] = locations.get(loc, 0) + 1
    
    embed = discord.Embed(
        title="📊 Offline Guard Team Status",
        description="Current team composition and capabilities",
        color=0x32cd32
    )
    
    embed.add_field(name="👥 Total Members", value=str(total_members), inline=True)
    embed.add_field(name="🛠️ Unique Skills", value=str(len(all_skills)), inline=True)
    embed.add_field(name="🌍 Locations", value=str(len(locations)), inline=True)
    
    if all_skills:
        embed.add_field(name="Skills Pool", value=", ".join(sorted(all_skills)), inline=False)
    
    if locations:
        location_str = "\n".join([f"{loc}: {count}" for loc, count in sorted(locations.items())])
        embed.add_field(name="Team Distribution", value=location_str, inline=False)
    
    await ctx.send(embed=embed)

if __name__ == "__main__":
    if BOT_TOKEN == 'your-bot-token-here':
        print("❌ Please set your Discord bot token in the DISCORD_BOT_TOKEN environment variable")
    else:
        bot.run(BOT_TOKEN)