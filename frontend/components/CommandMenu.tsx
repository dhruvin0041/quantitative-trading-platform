"use client"

import * as React from "react"
import {
  Moon,
  Sun,
  Laptop,
  Download,
  LayoutTemplate,
  Command as CommandIcon,
} from "lucide-react"

import {
  CommandDialog,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
  CommandSeparator,
  CommandShortcut,
} from "@/components/ui/command"
import { useTheme } from "next-themes"
import { toast } from "sonner"

export function CommandMenu() {
  const [open, setOpen] = React.useState(false)
  const { setTheme } = useTheme()

  React.useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if (e.key === "k" && e.altKey) {
        e.preventDefault()
        setOpen((open) => !open)
      }
    }

    const openCommand = () => setOpen(true)

    document.addEventListener("keydown", down)
    window.addEventListener("hydra-open-command", openCommand)
    
    return () => {
      document.removeEventListener("keydown", down)
      window.removeEventListener("hydra-open-command", openCommand)
    }
  }, [])

  const runCommand = React.useCallback((command: () => void) => {
    setOpen(false)
    // Wrap in try-catch to prevent component crashes on bad command execution
    try {
      command()
    } catch (err) {
      console.error("Command execution failed", err)
      toast.error("Failed to execute command.")
    }
  }, [])

  const toggleTheme = (newTheme: string) => {
    setTheme(newTheme)
    document.documentElement.setAttribute('data-theme', newTheme)
    localStorage.setItem('theme', newTheme)
  }

  return (
    <>
      <button
        onClick={() => setOpen(true)}
        className="flex items-center gap-2 px-3 py-1.5 text-[10px] uppercase tracking-widest font-black transition-all cursor-pointer bg-primary text-white border border-primary/50 shadow-md hover:bg-[var(--accent-hover)] dark:bg-white/5 dark:text-muted-foreground dark:border-white/10 dark:shadow-none dark:hover:bg-white/10 dark:hover:text-foreground rounded-md"
      >
        <CommandIcon className="w-3 h-3" />
        <span className="hidden sm:inline">Terminal Command</span>
        <kbd className="pointer-events-none inline-flex h-5 select-none items-center gap-1 rounded border border-white/20 bg-black/20 px-1.5 font-mono text-[10px] font-medium text-white dark:text-muted-foreground opacity-100">
          <span className="text-xs">⌘</span>K
        </kbd>
      </button>
      <CommandDialog open={open} onOpenChange={setOpen}>
        <CommandInput placeholder="Execute protocol..." className="font-mono" />
        <CommandList className="font-sans border-t border-border bg-popover/95 backdrop-blur-xl">
          <CommandEmpty className="py-6 text-center text-xs font-mono opacity-50 uppercase tracking-widest">Protocol not found.</CommandEmpty>
          <CommandGroup heading="Intelligence & Data">
            <CommandItem onSelect={() => runCommand(() => toast.info("Compiling analytics..."))} className="py-3">
              <Download className="mr-3 h-4 w-4 opacity-70" />
              <span className="text-sm font-medium text-foreground">Export Environment Snapshot</span>
              <CommandShortcut className="font-mono text-[10px]">⌘E</CommandShortcut>
            </CommandItem>
            <CommandItem onSelect={() => runCommand(() => toast.success("Matrix synchronized"))} className="py-3">
              <LayoutTemplate className="mr-3 h-4 w-4 opacity-70" />
              <span className="text-sm font-medium text-foreground">Synchronize Display Matrix</span>
              <CommandShortcut className="font-mono text-[10px]">⌘S</CommandShortcut>
            </CommandItem>
          </CommandGroup>
          <CommandSeparator className="bg-border" />
          <CommandGroup heading="System Parameters">
            <CommandItem onSelect={() => runCommand(() => toggleTheme("light"))} className="py-3">
              <Sun className="mr-3 h-4 w-4 opacity-70" />
              <span className="text-sm font-medium text-foreground">Protocol: LIGHT</span>
            </CommandItem>
            <CommandItem onSelect={() => runCommand(() => toggleTheme("dark"))} className="py-3">
              <Moon className="mr-3 h-4 w-4 opacity-70" />
              <span className="text-sm font-medium text-foreground">Protocol: DARK (OLED)</span>
            </CommandItem>
            <CommandItem onSelect={() => runCommand(() => toggleTheme("system"))} className="py-3">
              <Laptop className="mr-3 h-4 w-4 opacity-70" />
              <span className="text-sm font-medium text-foreground">Protocol: SYSTEM</span>
            </CommandItem>
          </CommandGroup>
        </CommandList>
      </CommandDialog>
    </>
  )
}


