"use client"

import * as React from "react"
import { Moon, Sun } from "lucide-react"
import { useTheme } from "next-themes"

import { Button } from "@/components/ui/button"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"

export function ThemeToggle() {
  const { theme, setTheme } = useTheme()
  const [mounted, setMounted] = React.useState(false)

  // Avoid hydration mismatch
  React.useEffect(() => {
    setMounted(true)
  }, [])

  const toggleTheme = (newTheme: string) => {
    setTheme(newTheme)
    document.documentElement.setAttribute('data-theme', newTheme)
    localStorage.setItem('theme', newTheme)
  }

  if (!mounted) return <div className="w-9 h-9" />

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="ghost" size="icon" className="rounded-full w-9 h-9 border border-border bg-secondary/50 hover:bg-secondary transition-all">
          <Sun className="h-[1.2rem] w-[1.2rem] rotate-0 scale-100 transition-all dark:-rotate-90 dark:scale-0 text-primary" />
          <Moon className="absolute h-[1.2rem] w-[1.2rem] rotate-90 scale-0 transition-all dark:rotate-0 dark:scale-100" />
          <span className="sr-only">Toggle theme</span>
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="font-sans bg-popover border-border">
        <DropdownMenuItem onClick={() => toggleTheme("light")} className="cursor-pointer hover:bg-secondary">
          Light (Institutional Orange)
        </DropdownMenuItem>
        <DropdownMenuItem onClick={() => toggleTheme("dark")} className="cursor-pointer hover:bg-secondary">
          Dark (OLED Black)
        </DropdownMenuItem>
        <DropdownMenuItem onClick={() => toggleTheme("system")} className="cursor-pointer hover:bg-secondary">
          Follow System Protocol
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
